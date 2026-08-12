#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
 
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <linux/fiemap.h>
#include <linux/nvme_ioctl.h>

#include "image.h"

#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "image_opencv.h"

/* OpenCV CAP_PROP_* values. image_opencv.h does not export the enum, and the
 * numeric values are stable across OpenCV 2.x-4.x. */
#define VMAP_CAP_PROP_FRAME_WIDTH   3
#define VMAP_CAP_PROP_FRAME_HEIGHT  4

/* NVMe logical block size assumed by the target when translating file
 * extents into LBAs. Must match the target's bdev block size * 8. */
#define NDP_BLOCK_SIZE       4096
 
/* Upper bound on file extents carried in one 0xC0 command. A file with more
 * extents than this is truncated, which would silently drop video data, so
 * the caller must check the warning printed by ndp_get_extents(). */
#define NDP_MAX_EXTENTS      128
 
/* The 0xC0 ioctl blocks until the target finishes decoding the whole clip,
 * so the timeout has to cover worst-case preprocessing, not just I/O. */
#define NDP_NVME_TIMEOUT_MS  120000

typedef struct {
    uint32_t count;
    uint64_t lba[NDP_MAX_EXTENTS];
    uint64_t blk[NDP_MAX_EXTENTS];   /* block count, in NDP_BLOCK_SIZE units */
} ndp_extent_table_t;

static int ndp_get_extents(const char *filepath, ndp_extent_table_t *out);
static int ndp_send_0xc0(int nvme_fd, const char *mp4_path,
                         int sample_rate, int scaler_sel, int jpeg_quality,
                         uint32_t *total_result_size_out);
static int ndp_send_0xc2(int nvme_fd, void *buf, uint32_t total_bytes);

/* Defined in image_opencv.cpp (cv::imdecode + cv::cvtColor + mat_to_image).
 * The signature must match that definition exactly: extern "C" only removes
 * name mangling, so a prototype mismatch is undefined behaviour. */
image ndp_jpeg_to_image(const unsigned char *buf, size_t len);

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path, int mAP_epochs)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *valid_images = option_find_str(options, "valid", train_images);
    char *backup_directory = option_find_str(options, "backup", "/backup/");


    network net_map;
    if (calc_map) {
        FILE* valid_file = fopen(valid_images, "r");
        if (!valid_file) {
            printf("\n Error: There is no %s file for mAP calculation!\n Don't use -map flag.\n Or set valid=%s in your %s file. \n", valid_images, train_images, datacfg);
            error("Error!", DARKNET_LOC);
        }
        else fclose(valid_file);

        cuda_set_device(gpus[0]);
        printf(" Prepare additional network for mAP calculation...\n");
        net_map = parse_network_cfg_custom(cfgfile, 1, 1);
        net_map.benchmark_layers = benchmark_layers;
        const int net_classes = net_map.layers[net_map.n - 1].classes;

        int k;  // free memory unnecessary arrays
        for (k = 0; k < net_map.n - 1; ++k) free_layer_custom(net_map.layers[k], 1);

        char *name_list = option_find_str(options, "names", "data/names.list");
        int names_size = 0;
        char **names = get_labels_custom(name_list, &names_size);
        if (net_classes != names_size) {
            printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
                name_list, names_size, net_classes, cfgfile);
        }
        free_ptrs((void**)names, net_map.layers[net_map.n - 1].classes);
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    float avg_contrastive_acc = 0;
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int k;
    for (k = 0; k < ngpus; ++k) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[k]);
#endif
        nets[k] = parse_network_cfg(cfgfile);
        nets[k].benchmark_layers = benchmark_layers;
        if (weightfile) {
            load_weights(&nets[k], weightfile);
        }
        if (clear) {
            *nets[k].seen = 0;
            *nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        error("Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64", DARKNET_LOC);
    }
    else if (actual_batch_size < 8) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }

    int save_after_iterations = option_find_int(options, "saveweights", (net.max_batches < 10000) ? 1000 : 10000 );  // configure when to write weights. Very useful for smaller datasets!
	int save_last_weights_after = option_find_int(options, "savelast", 100);
    printf("Weights are saved after: %d iterations. Last weights (*_last.weight) are stored every %d iterations. \n", save_after_iterations, save_last_weights_after );


    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            l = lk;
            printf(" Detection layer: %d - type = %d \n", k, l.type);
        }
    }

    int classes = l.classes;

    list *plist = get_paths(train_images);
    int train_images_num = plist->size;
    char **paths = (char **)list_to_array(plist);

    const int init_w = net.w;
    const int init_h = net.h;
    const int init_b = net.batch;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_iteration(net);
    iter_save_last = get_current_iteration(net);
    iter_map = get_current_iteration(net);
    float mean_average_precision = -1;
    float best_map = mean_average_precision;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = l.jitter;
    args.resize = l.resize;
    args.num_boxes = l.max_boxes;
    args.truth_size = l.truth_size;
    net.num_boxes = args.num_boxes;
    net.train_images_num = train_images_num;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.gaussian_noise = net.gaussian_noise;
    args.blur = net.blur;
    args.mixup = net.mixup;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.letter_box = net.letter_box;
    args.mosaic_bound = net.mosaic_bound;
    args.contrastive = net.contrastive;
    args.contrastive_jit_flip = net.contrastive_jit_flip;
    args.contrastive_color = net.contrastive_color;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;

#ifdef OPENCV
    //int num_threads = get_num_threads();
    //if(num_threads > 2) args.threads = get_num_threads() - 2;
    args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    //args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
    mat_cv* img = NULL;
    float max_img_loss = net.max_chart_loss;
    int number_of_lines = 100;
    int img_size = 1000;
    char windows_name[100];
    sprintf(windows_name, "chart_%s.png", base);
    img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
#endif    //OPENCV
    if (net.contrastive && args.threads > net.batch/2) args.threads = net.batch / 2;
    if (net.track) {
        args.track = net.track;
        args.augment_speed = net.augment_speed;
        if (net.sequential_subdivisions) args.threads = net.sequential_subdivisions * ngpus;
        else args.threads = net.subdivisions * ngpus;
        args.mini_batch = net.batch / net.time_steps;
        printf("\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = %d \n", net.batch, net.subdivisions, net.time_steps, args.mini_batch);
    }
    //printf(" imgs = %d \n", imgs);

    pthread_t load_thread = load_data(args);

    int count = 0;
    double time_remaining, avg_time = -1, alpha_time = 0.01;

    //while(i*imgs < N*120){
    while (get_current_iteration(net) < net.max_batches) {
        if (l.random && count++ % 10 == 0) {
            float rand_coef = 1.4;
            if (l.random != 1.0) rand_coef = l.random;
            printf("Resizing, random_coef = %.2f \n", rand_coef);
            float random_val = rand_scale(rand_coef);    // *x or /x
            int dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;
            int dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            int max_dim_w = roundl(rand_coef*init_w / net.resize_step + 1) * net.resize_step;
            int max_dim_h = roundl(rand_coef*init_h / net.resize_step + 1) * net.resize_step;

            // at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
            if (avg_loss < 0 || get_current_iteration(net) > net.max_batches - 100) {
                dim_w = max_dim_w;
                dim_h = max_dim_h;
            }

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;
            int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
            int new_dim_b = (int)(dim_b * 0.8);
            if (new_dim_b > init_b) dim_b = new_dim_b;

            args.w = dim_w;
            args.h = dim_h;

            int k;
            if (net.dynamic_minibatch) {
                for (k = 0; k < ngpus; ++k) {
                    (*nets[k].seen) = init_b * net.subdivisions * get_current_iteration(net); // remove this line, when you will save to weights-file both: seen & cur_iteration
                    nets[k].batch = dim_b;
                    int j;
                    for (j = 0; j < nets[k].n; ++j)
                        nets[k].layers[j].batch = dim_b;
                }
                net.batch = dim_b;
                imgs = net.batch * net.subdivisions * ngpus;
                args.n = imgs;
                printf("\n %d x %d  (batch = %d) \n", dim_w, dim_h, net.batch);
            }
            else
                printf("\n %d x %d \n", dim_w, dim_h);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for (k = 0; k < ngpus; ++k) {
                resize_network(nets + k, dim_w, dim_h);
            }
            net = nets[0];
        }
        double time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        if (net.track) {
            net.sequential_subdivisions = get_current_seq_subdivisions(net);
            args.threads = net.sequential_subdivisions * ngpus;
            printf(" sequential_subdivisions = %d, sequence = %d \n", net.sequential_subdivisions, get_sequence_value(net));
        }
        load_thread = load_data(args);
        //wait_key_cv(500);

        /*
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        if(!b.x) break;
        printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
        }
        image im = float_to_image(448, 448, 3, train.X.vals[10]);
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
        draw_bbox(im, b, 8, 1,0,0);
        }
        save_image(im, "truth11");
        */

        const double load_time = (what_time_is_it_now() - time);
        printf("Loaded: %lf seconds", load_time);
        if (load_time > 0.1 && avg_loss > 0) printf(" - performance bottleneck on CPU or Disk HDD/SSD");
        printf("\n");

        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            int wait_key = (dont_show) ? 0 : 1;
            loss = train_network_waitkey(net, train, wait_key);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        const int iteration = get_current_iteration(net);
        //i = get_current_batch(net);

        int calc_map_for_each = mAP_epochs * train_images_num / (net.batch * net.subdivisions);  // calculate mAP every mAP_epochs
        calc_map_for_each = fmax(calc_map_for_each, 100);
        int next_map_calc = iter_map + calc_map_for_each;
        next_map_calc = fmax(next_map_calc, net.burn_in);
        //next_map_calc = fmax(next_map_calc, 400);
        if (calc_map) {
            printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
            if (mean_average_precision > 0) printf("\n Last accuracy mAP@%0.2f = %2.2f %%, best = %2.2f %% ", iou_thresh, mean_average_precision * 100, best_map * 100);
        }

        printf("\033[H\033[J");
        if (mean_average_precision > 0.0) {
            printf("%d/%d: loss=%0.1f map=%0.2f best=%0.2f hours left=%0.1f\007", iteration, net.max_batches, loss, mean_average_precision, best_map, avg_time);
        }
        else {
            printf("%d/%d: loss=%0.1f hours left=%0.1f\007", iteration, net.max_batches, loss, avg_time);
        }

        if (net.cudnn_half) {
            if (iteration < net.burn_in * 3) fprintf(stderr, "\n Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net.burn_in);
            else fprintf(stderr, "\n Tensor Cores are used.\n");
            fflush(stderr);
        }
        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours left\n", iteration, loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), iteration*imgs, avg_time);
        fflush(stdout);

        int draw_precision = 0;
        if (calc_map && (iteration >= next_map_calc || iteration == net.max_batches)) {
            if (l.random) {
                printf("Resizing to initial size: %d x %d ", init_w, init_h);
                args.w = init_w;
                args.h = init_h;
                int k;
                if (net.dynamic_minibatch) {
                    for (k = 0; k < ngpus; ++k) {
                        for (k = 0; k < ngpus; ++k) {
                            nets[k].batch = init_b;
                            int j;
                            for (j = 0; j < nets[k].n; ++j)
                                nets[k].layers[j].batch = init_b;
                        }
                    }
                    net.batch = init_b;
                    imgs = init_b * net.subdivisions * ngpus;
                    args.n = imgs;
                    printf("\n %d x %d  (batch = %d) \n", init_w, init_h, init_b);
                }
                pthread_join(load_thread, 0);
                free_data(train);
                train = buffer;
                load_thread = load_data(args);
                for (k = 0; k < ngpus; ++k) {
                    resize_network(nets + k, init_w, init_h);
                }
                net = nets[0];
            }

            copy_weights_net(net, &net_map);

            // combine Training and Validation networks
            //network net_combined = combine_train_valid_networks(net, net_map);

            iter_map = iteration;
            mean_average_precision = validate_detector_map(datacfg, cfgfile, weightfile, thresh, iou_thresh, 0, net.letter_box, &net_map);// &net_combined);
            printf("\n mean_average_precision (mAP@%0.2f) = %f \n", iou_thresh, mean_average_precision);
            if (mean_average_precision >= best_map) {
                best_map = mean_average_precision;
                printf("New best mAP!\n");
                char buff[256];
                sprintf(buff, "%s/%s_best.weights", backup_directory, base);
                save_weights(net, buff);
            }

            draw_precision = 1;
        }
        time_remaining = ((net.max_batches - iteration) / ngpus)*(what_time_is_it_now() - time + load_time) / 60 / 60;
        // set initial value, even if resume training from 10000 iteration
        if (avg_time < 0) avg_time = time_remaining;
        else avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
#ifdef OPENCV
        if (net.contrastive) {
            float cur_con_acc = -1;
            for (k = 0; k < net.n; ++k)
                if (net.layers[k].type == CONTRASTIVE) cur_con_acc = *net.layers[k].loss;
            if (cur_con_acc >= 0) avg_contrastive_acc = avg_contrastive_acc*0.99 + cur_con_acc * 0.01;
            printf("  avg_contrastive_acc = %f \n", avg_contrastive_acc);
        }
        draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, net.max_batches, mean_average_precision, draw_precision, "mAP%", avg_contrastive_acc / 100, dont_show, mjpeg_port, avg_time);
#endif    // OPENCV

        if ( (iteration >= (iter_save + save_after_iterations) || iteration % save_after_iterations == 0) )
        {
            iter_save = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
            save_weights(net, buff);
        }

        if ( (save_after_iterations > save_last_weights_after) && (iteration >= (iter_save_last + save_last_weights_after) || (iteration % save_last_weights_after == 0 && iteration > 1))) {
            iter_save_last = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_last.weights", backup_directory, base);
            save_weights(net, buff);

            if (net.ema_alpha && is_ema_initialized(net)) {
                sprintf(buff, "%s/%s_ema.weights", backup_directory, base);
                save_weights_upto(net, buff, net.n, 1);
                printf(" EMA weights are saved to the file: %s \n", buff);
            }
        }
        free_data(train);
    }
#ifdef GPU
    if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    printf("If you want to train from the beginning, then use flag in the end of training command: -clear \n");

#ifdef OPENCV
    release_mat(&img);
    destroy_all_windows_cv();
#endif

    // free memory
    pthread_join(load_thread, 0);
    free_data(buffer);

    free_load_threads(&args);

    free(base);
    free(paths);
    free_list_contents(plist);
    free_list(plist);

    free_list_contents_kvp(options);
    free_list(options);

    for (k = 0; k < ngpus; ++k) free_network(nets[k]);
    free(nets);
    //free_network(net);

    if (calc_map) {
        net_map.n = 0;
        free_network(net_map);
    }
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c) p = c;
    return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    //int image_id = get_coco_image_id(image_path);
    char *p = basecfg(image_path);
    int image_id = atoi(p);
    for (i = 0; i < num_boxes; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > 0) {
                char buff[1024];
                sprintf(buff, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
                fprintf(fp, "%s", buff);
                //printf("%s", buff);
            }
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            int myclass = j;
            if (dets[i].prob[myclass] > 0) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[myclass],
                xmin, ymin, xmax, ymax);
        }
    }
}

static void print_kitti_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h, char *outfile, char *prefix)
{
    char *kitti_ids[] = { "car", "pedestrian", "cyclist" };
    FILE *fpd = 0;
    char buffd[1024];
    snprintf(buffd, 1024, "%s/%s/data/%s.txt", prefix, outfile, id);

    fpd = fopen(buffd, "w");
    int i, j;
    for (i = 0; i < total; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j)
        {
            //if (dets[i].prob[j]) fprintf(fpd, "%s 0 0 0 %f %f %f %f -1 -1 -1 -1 0 0 0 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
            if (dets[i].prob[j]) fprintf(fpd, "%s -1 -1 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
        }
    }
    fclose(fpd);
}

static void eliminate_bdd(char *buf, char *a)
{
    int n = 0;
    int i, k;
    for (i = 0; buf[i] != '\0'; i++)
    {
        if (buf[i] == a[n])
        {
            k = i;
            while (buf[i] == a[n])
            {
                if (a[++n] == '\0')
                {
                    for (k; buf[k + n] != '\0'; k++)
                    {
                        buf[k] = buf[k + n];
                    }
                    buf[k] = '\0';
                    break;
                }
                i++;
            }
            n = 0; i--;
        }
    }
}

static void get_bdd_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    eliminate_bdd(p, ".jpg");
    eliminate_bdd(p, "/");
    strcpy(filename, p);
}

static void print_bdd_detections(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    char *bdd_ids[] = { "bike" , "bus" , "car" , "motor" ,"person", "rider", "traffic light", "traffic sign", "train", "truck" };
    get_bdd_image_id(image_path);
    int i, j;

    for (i = 0; i < num_boxes; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx1 = xmin;
        float by1 = ymin;
        float bx2 = xmax;
        float by2 = ymax;

        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j])
            {
                fprintf(fp, "\t{\n\t\t\"name\":\"%s\",\n\t\t\"category\":\"%s\",\n\t\t\"bbox\":[%f, %f, %f, %f],\n\t\t\"score\":%f\n\t},\n", image_path, bdd_ids[j], bx1, by1, bx2, by2, dets[i].prob[j]);
            }
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];
    int k;
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            l = lk;
            printf(" Detection layer: %d - type = %d \n", k, l.type);
        }
    }
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    int bdd = 0;
    int kitti = 0;

    if (0 == strcmp(type, "coco")) {
        if (!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    }
    else if (0 == strcmp(type, "bdd")) {
        if (!outfile) outfile = "bdd_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        bdd = 1;
    }
    else if (0 == strcmp(type, "kitti")) {
        char buff2[1024];
        if (!outfile) outfile = "kitti_results";
        printf("%s\n", outfile);
        snprintf(buff, 1024, "%s/%s", prefix, outfile);
        int mkd = make_directory(buff, 0777);
        snprintf(buff2, 1024, "%s/%s/data", prefix, outfile);
        int mkd2 = make_directory(buff2, 0777);
        kitti = 1;
    }
    else if (0 == strcmp(type, "imagenet")) {
        if (!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    }
    else {
        if (!outfile) outfile = "comp4_det_test_";
        fps = (FILE**) xcalloc(classes, sizeof(FILE *));
        for (j = 0; j < classes; ++j) {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i = 0;
    int t;

    float thresh = .001;
    float nms = .6;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = IMAGE_DATA;
    const int letter_box = net.letter_box;
    if (letter_box) args.type = LETTERBOX_DATA;

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            char *path = paths[i + t - nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            if (coco) {
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            }
            else if (imagenet) {
                print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
            }
            else if (bdd) {
                print_bdd_detections(fp, path, dets, nboxes, classes, w, h);
            }
            else if (kitti) {
                print_kitti_detections(fps, id, dets, nboxes, classes, w, h, outfile, prefix);
            }
            else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }

            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    if (fps) {
        for (j = 0; j < classes; ++j) {
            fclose(fps[j]);
        }
        free(fps);
    }
    if (coco) {
#ifdef _WIN32
        fseek(fp, -3, SEEK_CUR);
#else
        fseek(fp, -2, SEEK_CUR);
#endif
        fprintf(fp, "\n]\n");
    }

    if (bdd) {
#ifdef _WIN32
        fseek(fp, -3, SEEK_CUR);
#else
        fseek(fp, -2, SEEK_CUR);
#endif
        fprintf(fp, "\n]\n");
        fclose(fp);
    }

    if (fp) fclose(fp);

    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)time(0) - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    srand(time(0));

    //list *plist = get_paths("data/coco_val_5k.list");
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    //layer l = net.layers[net.n - 1];

    int j, k;

    int m = plist->size;
    int i = 0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for (i = 0; i < m; ++i) {
        char *path = paths[i];
        image orig = load_image(path, 0, 0, net.c);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        int letterbox = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for (k = 0; k < nboxes; ++k) {
            if (dets[k].objectness > thresh) {
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
            float best_iou = 0;
            for (k = 0; k < nboxes; ++k) {
                float iou = box_iou(dets[k].bbox, t);
                if (dets[k].objectness > thresh && iou > best_iou) {
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if (best_iou > iou_thresh) {
                ++correct;
            }
        }
        //fprintf(stderr, " %s - %s - ", paths[i], labelpath);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
        free(truth);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(const box_prob *)pa;
    box_prob b = *(const box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}
 
/* One detection, retained across the whole run so that the PR curve can be
 * recomputed at several IoU thresholds without re-running inference. */
typedef struct {
    box   b;
    float p;
    int   class_id;
    int   image_index;
    float max_iou;           /* best IoU against same-class GT, 0 if none */
    int   best_truth_index;  /* global unique-truth index, -1 if none */
} vmap_det_t;
 
static int vmap_det_comparator(const void *pa, const void *pb)
{
    const vmap_det_t *a = (const vmap_det_t *)pa;
    const vmap_det_t *b = (const vmap_det_t *)pb;
    float diff = a->p - b->p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}
 
/*
 * Compute AP for every annotated class at one IoU threshold and return the
 * mean. Detections are consumed in descending-confidence order and each
 * ground-truth box may only be matched once, which is the standard greedy
 * assignment used by VOC and COCO.
 *
 * ap_out, when non-NULL, receives the per-class AP values.
 */
static double vmap_compute_map(const vmap_det_t *dets, int ndets,
                               int n_present,
                               const int *truth_count_present,
                               int unique_truth_count,
                               const int *cls_to_compact,
                               float iou_t, int map_points,
                               double *ap_out)
{
    typedef struct { double precision, recall; int tp, fp; } pr_t;
    int i, rank;
 
    if (ndets <= 0 || n_present <= 0) return 0.0;
 
    pr_t **pr = (pr_t **)xcalloc(n_present, sizeof(pr_t *));
    for (i = 0; i < n_present; ++i) pr[i] = (pr_t *)xcalloc(ndets, sizeof(pr_t));
 
    int *truth_flags = (int *)xcalloc(unique_truth_count > 0 ? unique_truth_count : 1,
                                      sizeof(int));
 
    for (rank = 0; rank < ndets; ++rank) {
        if (rank > 0) {
            for (i = 0; i < n_present; ++i) {
                pr[i][rank].tp = pr[i][rank - 1].tp;
                pr[i][rank].fp = pr[i][rank - 1].fp;
            }
        }
        int ci = cls_to_compact[dets[rank].class_id];
        if (ci >= 0) {
            int matched = (dets[rank].best_truth_index > -1 &&
                           dets[rank].max_iou > iou_t);
            if (matched && truth_flags[dets[rank].best_truth_index] == 0) {
                truth_flags[dets[rank].best_truth_index] = 1;
                pr[ci][rank].tp++;
            } else {
                pr[ci][rank].fp++;
            }
        }
        for (i = 0; i < n_present; ++i) {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_count_present[i] - tp;
            pr[i][rank].precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
            pr[i][rank].recall    = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
        }
    }
    free(truth_flags);
 
    double sum_ap = 0;
    for (i = 0; i < n_present; ++i) {
        double avg_precision = 0;
        if (map_points == 0) {
            /* Area under the interpolated PR curve (VOC 2010-2012, ImageNet) */
            double last_recall    = pr[i][ndets - 1].recall;
            double last_precision = pr[i][ndets - 1].precision;
            for (rank = ndets - 2; rank >= 0; --rank) {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;
                if (pr[i][rank].precision > last_precision)
                    last_precision = pr[i][rank].precision;
                avg_precision += delta_recall * last_precision;
            }
            avg_precision += last_recall * last_precision;
        } else {
            /* Fixed recall points (COCO uses 101, VOC 2007 uses 11) */
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points - 1);
                double cur_precision = 0;
                for (rank = 0; rank < ndets; ++rank) {
                    if (pr[i][rank].recall >= cur_recall &&
                        pr[i][rank].precision > cur_precision)
                        cur_precision = pr[i][rank].precision;
                }
                avg_precision += cur_precision;
            }
            avg_precision /= map_points;
        }
        if (ap_out) ap_out[i] = avg_precision;
        sum_ap += avg_precision;
    }
 
    for (i = 0; i < n_present; ++i) free(pr[i]);
    free(pr);
 
    return sum_ap / n_present;
}

/*
 * Detection accuracy evaluation over a video file.
 *
 * Both the host-centric baseline and the NDP pipeline are evaluated by this
 * single function: only the frame acquisition path differs, while ground-truth
 * loading, matching and AP computation are shared. Keeping them on one code
 * path is what makes the two conditions directly comparable, since the
 * differences being measured are on the order of one AP point.
 *
 *   use_ndp = 0   frames are decoded on the host with OpenCV and letterboxed
 *                 by darknet's letterbox_image()
 *   use_ndp = 1   the target decodes, samples, letterboxes and JPEG-encodes;
 *                 the host only decompresses
 *
 * Ground truth is read from <label_dir>/%06d.txt in YOLO format, normalised
 * against the source resolution. The file index is (decoded frame index +
 * label_base); MOT-style datasets whose first frame is 000001.jpg need
 * label_base = 1.
 */
float validate_video_map(char *datacfg, char *cfgfile, char *weightfile,
                         char *video_path, char *label_dir, char *nvme_dev,
                         int use_ndp, int sample_rate, int label_base,
                         float thresh_calc_avg_iou, const float iou_thresh,
                         const int map_points, int coco_range,
                         int scaler_sel, int jpeg_quality)
{
    int i, j;
 
    if (!video_path) { fprintf(stderr, "[MAP] -file <video> is required\n"); return 0; }
    if (!label_dir)  { fprintf(stderr, "[MAP] -labels <dir> is required\n"); return 0; }
    if (sample_rate < 1) sample_rate = 1;
 
    /* ---- network ---- */
    list *options    = read_data_cfg(datacfg);
    char *name_list  = option_find_str(options, "names", "data/names.list");
    int   names_size = 0;
    char **names     = get_labels_custom(name_list, &names_size);
 
    network net = parse_network_cfg_custom(cfgfile, 1, 1);   /* batch=1 */
    if (weightfile) load_weights(&net, weightfile);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);
 
    layer l = net.layers[net.n - 1];
    for (i = 0; i < net.n; ++i) {
        layer lk = net.layers[i];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) l = lk;
    }
    const int classes = l.classes;
    if (classes != names_size)
        fprintf(stderr, "[MAP] Warning: cfg classes=%d but names=%d\n", classes, names_size);
 
    /* ---- source geometry ---- */
    cap_cv *cap = get_capture_video_stream(video_path);
    if (!cap) { fprintf(stderr, "[MAP] Cannot open video: %s\n", video_path); return 0; }
 
    int orig_w = (int)get_capture_property_cv(cap, VMAP_CAP_PROP_FRAME_WIDTH);
    int orig_h = (int)get_capture_property_cv(cap, VMAP_CAP_PROP_FRAME_HEIGHT);
    int total_frames = (int)get_capture_frame_count_cv(cap);
    if (orig_w <= 0 || orig_h <= 0) {
        fprintf(stderr, "[MAP] Invalid video dimensions (%dx%d)\n", orig_w, orig_h);
        release_capture(cap);
        return 0;
    }
 
    printf("[MAP][VERIFY] video=%s  orig=%dx%d  cap_frame_count=%d  "
           "sample_rate=1/%d  mode=%s  scaler_sel=%d  jpeg_q=%d\n",
           video_path, orig_w, orig_h, total_frames, sample_rate,
           use_ndp ? "NDP" : "BASELINE",
           use_ndp ? scaler_sel : -1,
           use_ndp ? (jpeg_quality ? jpeg_quality : 85) : -1);
    {
        /* Letterbox geometry darknet will assume when inverting box
         * coordinates. The target logs the same four numbers as
         * [NDP-verify2]; if they disagree, NDP boxes carry a systematic
         * offset and the comparison is invalid. */
        int lw, lh;
        if (((float)net.w / orig_w) < ((float)net.h / orig_h)) {
            lw = net.w; lh = (orig_h * net.w) / orig_w;
        } else {
            lh = net.h; lw = (orig_w * net.h) / orig_h;
        }
        printf("[MAP][VERIFY] darknet letterbox: scaled=%dx%d dx=%d dy=%d "
               "canvas=%dx%d\n", lw, lh, (net.w - lw) / 2, (net.h - lh) / 2,
               net.w, net.h);
    }
 
    /* ---- NDP mode: fetch the whole preprocessed result up front ---- */
    int       nvme_fd    = -1;
    uint8_t  *result_buf = NULL;
    const uint32_t *jpeg_sizes = NULL;
    const uint8_t  *blob_base  = NULL;
    uint32_t  num_ndp_frames   = 0;
    size_t    blob_off         = 0;
 
    if (use_ndp) {
        nvme_fd = open(nvme_dev, O_RDWR);
        if (nvme_fd < 0) {
            fprintf(stderr, "[MAP] Cannot open NVMe device %s: %s\n",
                    nvme_dev, strerror(errno));
            release_capture(cap); return 0;
        }
        uint32_t total_result_size = 0;
        if (ndp_send_0xc0(nvme_fd, video_path, sample_rate,
                          scaler_sel, jpeg_quality, &total_result_size) < 0 ||
            total_result_size == 0) {
            fprintf(stderr, "[MAP] 0xC0 failed\n");
            close(nvme_fd); release_capture(cap); return 0;
        }
        size_t alloc_bytes = ((size_t)total_result_size + 4095) & ~(size_t)4095;
        result_buf = (uint8_t *)aligned_alloc(4096, alloc_bytes);
        if (!result_buf) {
            fprintf(stderr, "[MAP] alloc %zu bytes failed\n", alloc_bytes);
            close(nvme_fd); release_capture(cap); return 0;
        }
        memset(result_buf, 0, alloc_bytes);
        if (ndp_send_0xc2(nvme_fd, result_buf, total_result_size) < 0) {
            fprintf(stderr, "[MAP] 0xC2 failed\n");
            free(result_buf); close(nvme_fd); release_capture(cap); return 0;
        }
        memcpy(&num_ndp_frames, result_buf, sizeof(uint32_t));
        size_t header_bytes = (size_t)(1 + num_ndp_frames) * sizeof(uint32_t);
        if (num_ndp_frames == 0 || header_bytes > (size_t)total_result_size) {
            fprintf(stderr, "[MAP] Invalid result header (frames=%u)\n", num_ndp_frames);
            free(result_buf); close(nvme_fd); release_capture(cap); return 0;
        }
        jpeg_sizes = (const uint32_t *)(result_buf + sizeof(uint32_t));
        blob_base  = result_buf + header_bytes;
 
        /* Frame alignment guard. The host derives each frame's ground-truth
         * index as image_index * sample_rate, so a single dropped frame on the
         * target would shift every subsequent label without any other symptom. */
        if (total_frames > 0) {
            uint32_t expected = (uint32_t)((total_frames - 1) / sample_rate) + 1;
            printf("[MAP][VERIFY] NDP frames=%u, expected=%u\n",
                   num_ndp_frames, expected);
            if (num_ndp_frames != expected) {
                fprintf(stderr,
                    "[MAP] *** FRAME ALIGNMENT MISMATCH *** ground-truth indices "
                    "would be wrong. Aborting.\n"
                    "      Cross-check ffprobe -count_frames against the target's "
                    "[NDP-verify] decoded_frames log.\n");
                free(result_buf); close(nvme_fd); release_capture(cap); return 0;
            }
        }
    }
 
    /* ---- accumulators ---- */
 
    /* Fixed low threshold: AP integration needs the low-confidence tail of the
     * PR curve, so this must not be replaced by the user-facing -thresh. */
    const float thresh = .005f;
    const float nms    = .45f;
 
    int cap_dets = 4096;
    vmap_det_t *dets_all = (vmap_det_t *)xcalloc(cap_dets, sizeof(vmap_det_t));
    int ndets = 0;
    int unique_truth_count = 0;
 
    int   *truth_classes_count     = (int *)xcalloc(classes, sizeof(int));
    float *avg_iou_per_class       = (float *)xcalloc(classes, sizeof(float));
    int   *tp_for_thresh_per_class = (int *)xcalloc(classes, sizeof(int));
    int   *fp_for_thresh_per_class = (int *)xcalloc(classes, sizeof(int));
 
    float avg_iou = 0;
    int image_index = 0;
    int missing_labels = 0;
 
    /* Declared before the first goto so that cleanup can free them */
    int *cls_to_compact = (int *)xcalloc(classes, sizeof(int));
    int *present_cls = NULL, *truth_count_present = NULL;
    double *ap_primary = NULL;
    int n_present = 0;
    double mean_ap = 0, mean_ap_coco = 0;
 
    time_t start = time(0);
 
    /* ---- frame loop ---- */
    int decode_idx = 0;
    for (;;) {
        image im_input;
        im_input.data = NULL; im_input.w = 0; im_input.h = 0; im_input.c = 0;
 
        if (use_ndp) {
            if ((uint32_t)image_index >= num_ndp_frames) break;
            im_input = ndp_jpeg_to_image(blob_base + blob_off, jpeg_sizes[image_index]);
            blob_off  += jpeg_sizes[image_index];
            decode_idx = image_index * sample_rate;
        } else {
            /* Same pixel operations as validate_detector_map()'s
             * LETTERBOX_DATA path; only the source differs. */
            mat_cv *in_img = NULL;
            im_input = get_image_from_stream_letterbox(cap, net.w, net.h,
                                                       net.c, &in_img, 0);
            if (in_img) release_mat(&in_img);
            if (!im_input.data || im_input.w == 0) break;
        }
 
        if (!im_input.data || im_input.w != net.w || im_input.h != net.h) {
            fprintf(stderr, "[MAP] frame %d: bad image (%dx%d), skipped\n",
                    decode_idx, im_input.w, im_input.h);
            if (im_input.data) free_image(im_input);
            ++image_index;
            if (!use_ndp) {
                for (i = 1; i < sample_rate; ++i) consume_frame(cap);
                decode_idx += sample_rate;
                if (total_frames > 0 && decode_idx >= total_frames) break;
            }
            continue;
        }
 
        network_predict(net, im_input.data);
 
        int nboxes = 0;
        /* Passing the source resolution rather than the 416x416 input size
         * makes correct_yolo_boxes() undo the letterbox transform, so boxes
         * come back in source-frame normalised coordinates and can be matched
         * against unmodified ground truth. */
        detection *dets = get_network_boxes(&net, orig_w, orig_h,
                                            thresh, 0.0f, 0, 1, &nboxes, 1);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
 
        char labelpath[4096];
        snprintf(labelpath, sizeof(labelpath), "%s/%06d.txt",
                 label_dir, decode_idx + label_base);
 
        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        if (num_labels == 0) ++missing_labels;
        for (j = 0; j < num_labels; ++j)
            if (truth[j].id >= 0 && truth[j].id < classes)
                truth_classes_count[truth[j].id]++;
 
        const int checkpoint = ndets;
 
        for (i = 0; i < nboxes; ++i) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                float prob = dets[i].prob[class_id];
                if (prob <= 0) continue;
 
                if (ndets >= cap_dets) {
                    cap_dets *= 2;
                    dets_all = (vmap_det_t *)xrealloc(dets_all,
                                    cap_dets * sizeof(vmap_det_t));
                }
 
                /* Store the best IoU rather than a hard match decision, so the
                 * same detection set can be re-scored at other thresholds. */
                float max_iou = 0; int truth_index = -1;
                for (j = 0; j < num_labels; ++j) {
                    if (class_id != truth[j].id) continue;
                    box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                    float cur = box_iou(dets[i].bbox, t);
                    if (cur > max_iou) { max_iou = cur; truth_index = unique_truth_count + j; }
                }
 
                dets_all[ndets].b                = dets[i].bbox;
                dets_all[ndets].p                = prob;
                dets_all[ndets].class_id         = class_id;
                dets_all[ndets].image_index      = image_index;
                dets_all[ndets].max_iou          = max_iou;
                dets_all[ndets].best_truth_index = (max_iou > 0) ? truth_index : -1;
                ndets++;
 
                /* Precision / recall / F1 statistics at the reporting
                 * threshold, independent of the AP integration above. */
                if (prob > thresh_calc_avg_iou) {
                    int z, found = 0;
                    int hit = (max_iou > iou_thresh && truth_index > -1);
                    for (z = checkpoint; z < ndets - 1; ++z)
                        if (dets_all[z].best_truth_index == truth_index &&
                            dets_all[z].max_iou > iou_thresh) { found = 1; break; }
                    if (hit && found == 0) {
                        avg_iou += max_iou;
                        avg_iou_per_class[class_id] += max_iou;
                        tp_for_thresh_per_class[class_id]++;
                    } else {
                        fp_for_thresh_per_class[class_id]++;
                    }
                }
            }
        }
        unique_truth_count += num_labels;
 
        if ((image_index % 50) == 0)
            fprintf(stderr, "\r[MAP] eval frame %d (video idx %d)   ",
                    image_index, decode_idx);
 
        free_detections(dets, nboxes);
        free(truth);
        free_image(im_input);
 
        ++image_index;
 
        if (!use_ndp) {
            for (i = 1; i < sample_rate; ++i) consume_frame(cap);
            decode_idx += sample_rate;
            if (total_frames > 0 && decode_idx >= total_frames) break;
        }
    }
    fprintf(stderr, "\n");
 
    const int num_eval_frames = image_index;
 
    /* ---- restrict scoring to classes that actually occur in the labels ----
     * Averaging over all 80 COCO classes would divide by mostly-empty entries
     * and make the reported mAP meaningless on a single-class dataset. */
    for (i = 0; i < classes; ++i) cls_to_compact[i] = -1;
    for (i = 0; i < classes; ++i) if (truth_classes_count[i] > 0) n_present++;
 
    if (n_present == 0) {
        fprintf(stderr, "[MAP] No ground-truth labels found under %s "
                        "(checked %d frames, %d missing). Aborting.\n",
                label_dir, num_eval_frames, missing_labels);
        goto cleanup;
    }
 
    present_cls         = (int *)xcalloc(n_present, sizeof(int));
    truth_count_present = (int *)xcalloc(n_present, sizeof(int));
    {
        int k = 0;
        for (i = 0; i < classes; ++i) {
            if (truth_classes_count[i] > 0) {
                cls_to_compact[i]      = k;
                present_cls[k]         = i;
                truth_count_present[k] = truth_classes_count[i];
                k++;
            }
        }
    }
 
    /* Detections of unannotated classes cannot affect any annotated class's
     * AP, so dropping them here only saves memory and sort time. */
    {
        int w = 0;
        for (i = 0; i < ndets; ++i)
            if (cls_to_compact[dets_all[i].class_id] >= 0) dets_all[w++] = dets_all[i];
        printf("[MAP] detections %d -> %d after class filtering "
               "(%d annotated class(es))\n", ndets, w, n_present);
        ndets = w;
    }
 
    qsort(dets_all, ndets, sizeof(vmap_det_t), vmap_det_comparator);
 
    /* ---- report ---- */
    {
        /* Precision and recall are aggregated over annotated classes only;
         * mixing in false positives from unannotated classes would understate
         * precision for the classes actually being evaluated. */
        int tp_present = 0, fp_present = 0;
        for (i = 0; i < n_present; ++i) {
            int c = present_cls[i];
            tp_present += tp_for_thresh_per_class[c];
            fp_present += fp_for_thresh_per_class[c];
        }
        if ((tp_present + fp_present) > 0) avg_iou /= (tp_present + fp_present);
 
        printf("\n eval frames = %d, detections = %d, unique GT boxes = %d, "
               "frames without label file = %d\n",
               num_eval_frames, ndets, unique_truth_count, missing_labels);
 
        ap_primary = (double *)xcalloc(n_present, sizeof(double));
        mean_ap = vmap_compute_map(dets_all, ndets, n_present,
                                   truth_count_present, unique_truth_count,
                                   cls_to_compact, iou_thresh, map_points,
                                   ap_primary);
 
        printf("\n--- per-class AP @ IoU=%0.2f ---\n", iou_thresh);
        for (i = 0; i < n_present; ++i) {
            int c = present_cls[i];
            printf("class_id = %d, name = %s, ap = %2.2f%%   \t"
                   "(TP = %d, FP = %d, GT = %d)\n",
                   c, names[c], ap_primary[i] * 100,
                   tp_for_thresh_per_class[c], fp_for_thresh_per_class[c],
                   truth_classes_count[c]);
        }
 
        const float cur_precision = (float)tp_present /
                                    ((float)tp_present + (float)fp_present);
        const float cur_recall    = (float)tp_present /
                                    ((float)tp_present +
                                     (float)(unique_truth_count - tp_present));
        const float f1_score = 2.F * cur_precision * cur_recall /
                               (cur_precision + cur_recall);
        printf("\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, "
               "F1-score = %1.2f\n",
               thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);
        printf(" for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, "
               "average IoU = %2.2f %%\n",
               thresh_calc_avg_iou, tp_present, fp_present,
               unique_truth_count - tp_present, avg_iou * 100);
    }
 
    printf("\n [%s] mAP@%0.2f = %2.2f %%   (%s, %d class(es))\n",
           use_ndp ? "NDP" : "BASELINE", iou_thresh, mean_ap * 100,
           map_points ? "recall-points" : "AUC", n_present);
 
    /* COCO-style average over IoU 0.50:0.05:0.95. Inference is not repeated;
     * only the PR curve is rebuilt from the stored per-detection IoUs. */
    if (coco_range) {
        printf("\n--- mAP over IoU=0.50:0.05:0.95 ---\n");
        double sum = 0;
        for (i = 0; i < 10; ++i) {
            float t = 0.50f + 0.05f * i;
            double m = vmap_compute_map(dets_all, ndets, n_present,
                                        truth_count_present, unique_truth_count,
                                        cls_to_compact, t, map_points, NULL);
            printf("   IoU=%0.2f : mAP = %2.2f %%\n", t, m * 100);
            sum += m;
        }
        mean_ap_coco = sum / 10.0;
        printf(" [%s] mAP@[.50:.95] = %2.2f %%\n",
               use_ndp ? "NDP" : "BASELINE", mean_ap_coco * 100);
    }
 
    fprintf(stderr, "Total Detection Time: %d Seconds\n", (int)(time(0) - start));
 
cleanup:
    if (ap_primary)          free(ap_primary);
    if (present_cls)         free(present_cls);
    if (truth_count_present) free(truth_count_present);
    free(cls_to_compact);
    free(dets_all);
    free(truth_classes_count);
    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);
 
    if (result_buf) free(result_buf);
    if (nvme_fd >= 0) close(nvme_fd);
    release_capture(cap);
 
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_network(net);
 
    return (float)(coco_range ? mean_ap_coco : mean_ap);
}

float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);
    //char *mapf = option_find_str(options, "map", 0);
    //int *map = 0;
    //if (mapf) map = read_map(mapf);
    FILE* reinforcement_fd = NULL;

    network net;
    //int initial_batch;
    if (existing_net) {
        char *train_images = option_find_str(options, "train", "data/train.txt");
        valid_images = option_find_str(options, "valid", train_images);
        net = *existing_net;
        remember_network_recurrent_state(*existing_net);
        free_network_recurrent_state(*existing_net);
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        error("Error!", DARKNET_LOC);
    }
    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    list *plist_dif = NULL;
    char **paths_dif = NULL;
    if (difficult_valid_images) {
        plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }

    layer l = net.layers[net.n - 1];
    int k;
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            l = lk;
            printf(" Detection layer: %d - type = %d \n", k, l.type);
        }
    }
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    const float thresh = .005;
    const float nms = .45;
    //const float iou_thresh = 0.5;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    letter_box = net.letter_box;
    if (letter_box) args.type = LETTERBOX_DATA;
    else args.type = IMAGE_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob* detections = (box_prob*)xcalloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int* truth_classes_count = (int*)xcalloc(classes, sizeof(int));

    // For multi-class precision and recall computation
    float *avg_iou_per_class = (float*)xcalloc(classes, sizeof(float));
    int *tp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));
    int *fp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "\r%d", i);
        for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && (i + t) < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);

            int nboxes = 0;
            float hier_thresh = 0;
            detection *dets;
            if (args.type == LETTERBOX_DATA) {
                dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
            }
            else {
                dets = get_network_boxes(&net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes, letter_box);
            }
            //detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letter_box); // for letter_box=1
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            //if (l.embedding_size) set_track_id(dets, nboxes, thresh, l.sim_thresh, l.track_ciou_norm, l.track_history_size, l.dets_for_track, l.dets_for_show);

            char labelpath[4096];
            replace_image_to_label(path, labelpath);
            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);
            int j;
            for (j = 0; j < num_labels; ++j) {
                truth_classes_count[truth[j].id]++;
            }

            // difficult
            box_label *truth_dif = NULL;
            int num_labels_dif = 0;
            if (paths_dif)
            {
                char *path_dif = paths_dif[image_index];

                char labelpath_dif[4096];
                replace_image_to_label(path_dif, labelpath_dif);

                truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
            }

            const int checkpoint_detections_count = detections_count;

            int i;
            for (i = 0; i < nboxes; ++i) {

                int class_id;
                for (class_id = 0; class_id < classes; ++class_id) {
                    float prob = dets[i].prob[class_id];
                    if (prob > 0) {
                        detections_count++;
                        detections = (box_prob*)xrealloc(detections, detections_count * sizeof(box_prob));
                        detections[detections_count - 1].b = dets[i].bbox;
                        detections[detections_count - 1].p = prob;
                        detections[detections_count - 1].image_index = image_index;
                        detections[detections_count - 1].class_id = class_id;
                        detections[detections_count - 1].truth_flag = 0;
                        detections[detections_count - 1].unique_truth_index = -1;

                        int truth_index = -1;
                        float max_iou = 0;
                        for (j = 0; j < num_labels; ++j)
                        {
                            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                            //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                            //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                            float current_iou = box_iou(dets[i].bbox, t);
                            if (current_iou > iou_thresh && class_id == truth[j].id) {
                                if (current_iou > max_iou) {
                                    max_iou = current_iou;
                                    truth_index = unique_truth_count + j;
                                }
                            }
                        }

                        // best IoU
                        if (truth_index > -1) {
                            detections[detections_count - 1].truth_flag = 1;
                            detections[detections_count - 1].unique_truth_index = truth_index;
                        }
                        else {
                            // if object is difficult then remove detection
                            for (j = 0; j < num_labels_dif; ++j) {
                                box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
                                float current_iou = box_iou(dets[i].bbox, t);
                                if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
                                    --detections_count;
                                    break;
                                }
                            }
                        }

                        // calc avg IoU, true-positives, false-positives for required Threshold
                        if (prob > thresh_calc_avg_iou) {
                            int z, found = 0;
                            for (z = checkpoint_detections_count; z < detections_count - 1; ++z) {
                                if (detections[z].unique_truth_index == truth_index) {
                                    found = 1; break;
                                }
                            }

                            if (truth_index > -1 && found == 0) {
                                avg_iou += max_iou;
                                ++tp_for_thresh;
                                avg_iou_per_class[class_id] += max_iou;
                                tp_for_thresh_per_class[class_id]++;
                            }
                            else{
                                fp_for_thresh++;
                                fp_for_thresh_per_class[class_id]++;
                            }
                        }
                    }
                }
            }

            unique_truth_count += num_labels;

            //static int previous_errors = 0;
            //int total_errors = fp_for_thresh + (unique_truth_count - tp_for_thresh);
            //int errors_in_this_image = total_errors - previous_errors;
            //previous_errors = total_errors;
            //if(reinforcement_fd == NULL) reinforcement_fd = fopen("reinforcement.txt", "wb");
            //char buff[1000];
            //sprintf(buff, "%s\n", path);
            //if(errors_in_this_image > 0) fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

            free_detections(dets, nboxes);
            free(truth);
            free(truth_dif);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }

    //for (t = 0; t < nthreads; ++t) {
    //    pthread_join(thr[t], 0);
    //}

    if ((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

    int class_id;
    for(class_id = 0; class_id < classes; class_id++){
        if ((tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]) > 0)
            avg_iou_per_class[class_id] = avg_iou_per_class[class_id] / (tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]);
    }

    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double prob;
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t** pr = (pr_t**)xcalloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t*)xcalloc(detections_count, sizeof(pr_t));
    }
    printf("\n detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int* detection_per_class_count = (int*)xcalloc(classes, sizeof(int));
    for (j = 0; j < detections_count; ++j) {
        detection_per_class_count[detections[j].class_id]++;
    }

    int* truth_flags = (int*)xcalloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        pr[d.class_id][rank].prob = d.p;
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else
                pr[d.class_id][rank].fp++;
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                    printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, detection_per_class_count[i], tp+fp, tp, fp);
            }
        }
    }

    free(truth_flags);

    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;

        // MS COCO - uses 101-Recall-points on PR-chart.
        // PascalVOC2007 - uses 11-Recall-points on PR-chart.
        // PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
        // ImageNet - uses Area-Under-Curve on PR-chart.

        // correct mAP calculation: ImageNet, PascalVOC 2010-2012
        if (map_points == 0)
        {
            double last_recall = pr[i][detections_count - 1].recall;
            double last_precision = pr[i][detections_count - 1].precision;
            for (rank = detections_count - 2; rank >= 0; --rank)
            {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) {
                    last_precision = pr[i][rank].precision;
                }

                avg_precision += delta_recall * last_precision;
            }
            //add remaining area of PR curve when recall isn't 0 at rank-1
            double delta_recall = last_recall - 0;
            avg_precision += delta_recall * last_precision;
        }
        // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        else
        {
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points-1);
                double cur_precision = 0;
                double cur_prob = 0;
                for (rank = 0; rank < detections_count; ++rank)
                {
                    if (pr[i][rank].recall >= cur_recall) {    // > or >=
                        if (pr[i][rank].precision > cur_precision) {
                            cur_precision = pr[i][rank].precision;
                            cur_prob = pr[i][rank].prob;
                        }
                    }
                }
                //printf("class_id = %d, point = %d, cur_prob = %.4f, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_prob, cur_recall, cur_precision);

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        printf("class_id = %d, name = %s, ap = %2.2f%%   \t (TP = %d, FP = %d) \n",
            i, names[i], avg_precision * 100, tp_for_thresh_per_class[i], fp_for_thresh_per_class[i]);

        float class_precision = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
        float class_recall = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
        //printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n", class_precision, class_recall, avg_iou_per_class[i]);

        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
    if (map_points) printf("used %d Recall-points \n", map_points);
    else printf("used Area-Under-Curve for each unique Recall \n");

    printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh, mean_average_precision, mean_average_precision * 100);

    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);
    free(paths);
    free(paths_dif);
    free_list_contents(plist);
    free_list(plist);
    if (plist_dif) {
        free_list_contents(plist_dif);
        free_list(plist_dif);
    }
    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);

    fprintf(stderr, "Total Detection Time: %d Seconds\n", (int)(time(0) - start));
    printf("\nSet -points flag:\n");
    printf(" `-points 101` for MS COCO \n");
    printf(" `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) \n");
    printf(" `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset\n");
    if (reinforcement_fd != NULL) fclose(reinforcement_fd);

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    if (existing_net) {
        //set_batch_network(&net, initial_batch);
        //free_network_recurrent_state(*existing_net);
        restore_network_recurrent_state(*existing_net);
        //randomize_network_recurrent_state(*existing_net);
    }
    else {
        free_network(net);
    }
    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    return mean_average_precision;
}

typedef struct {
    float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
    anchors_t a = *(const anchors_t *)pa;
    anchors_t b = *(const anchors_t *)pb;
    float diff = b.w*b.h - a.w*a.h;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int anchors_data_comparator(const float **pa, const float **pb)
{
    float *a = (float *)*pa;
    float *b = (float *)*pb;
    float diff = b[0] * b[1] - a[0] * a[1];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}


void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
    printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
    if (width < 0 || height < 0) {
        printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
        printf("Error: set width and height \n");
        return;
    }

    //float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
    float* rel_width_height_array = (float*)xcalloc(1000, sizeof(float));


    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    list *plist = get_paths(train_images);
    int number_of_images = plist->size;
    char **paths = (char **)list_to_array(plist);

    int classes = option_find_int(options, "classes", 1);
    int* counter_per_class = (int*)xcalloc(classes, sizeof(int));

    srand(time(0));
    int number_of_boxes = 0;
    printf(" read labels from %d images \n", number_of_images);

    int i, j;
    for (i = 0; i < number_of_images; ++i) {
        char *path = paths[i];
        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        //printf(" new path: %s \n", labelpath);
        char *buff = (char*)xcalloc(6144, sizeof(char));
        for (j = 0; j < num_labels; ++j)
        {
            if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
                truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0)
            {
                printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                system(buff);
            }
            if (truth[j].id >= classes) {
                classes = truth[j].id + 1;
                counter_per_class = (int*)xrealloc(counter_per_class, classes * sizeof(int));
            }
            counter_per_class[truth[j].id]++;

            number_of_boxes++;
            rel_width_height_array = (float*)xrealloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));

            rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
            rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
            printf("\r loaded \t image: %d \t box: %d", i + 1, number_of_boxes);
        }
        free(buff);
        free(truth);
    }
    printf("\n all loaded. \n");
    printf("\n calculating k-means++ ...");

    matrix boxes_data;
    model anchors_data;
    boxes_data = make_matrix(number_of_boxes, 2);

    printf("\n");
    for (i = 0; i < number_of_boxes; ++i) {
        boxes_data.vals[i][0] = rel_width_height_array[i * 2];
        boxes_data.vals[i][1] = rel_width_height_array[i * 2 + 1];
        //if (w > 410 || h > 410) printf("i:%d,  w = %f, h = %f \n", i, w, h);
    }

    // Is used: distance(box, centroid) = 1 - IoU(box, centroid)

    // K-means
    anchors_data = do_kmeans(boxes_data, num_of_clusters);

    qsort((void*)anchors_data.centers.vals, num_of_clusters, 2 * sizeof(float), (__compar_fn_t)anchors_data_comparator);

    //gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
    //float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

    printf("\n");
    float avg_iou = 0;
    for (i = 0; i < number_of_boxes; ++i) {
        float box_w = rel_width_height_array[i * 2]; //points->data.fl[i * 2];
        float box_h = rel_width_height_array[i * 2 + 1]; //points->data.fl[i * 2 + 1];
                                                         //int cluster_idx = labels->data.i[i];
        int cluster_idx = 0;
        float min_dist = FLT_MAX;
        float best_iou = 0;
        for (j = 0; j < num_of_clusters; ++j) {
            float anchor_w = anchors_data.centers.vals[j][0];   // centers->data.fl[j * 2];
            float anchor_h = anchors_data.centers.vals[j][1];   // centers->data.fl[j * 2 + 1];
            float min_w = (box_w < anchor_w) ? box_w : anchor_w;
            float min_h = (box_h < anchor_h) ? box_h : anchor_h;
            float box_intersect = min_w*min_h;
            float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
            float iou = box_intersect / box_union;
            float distance = 1 - iou;
            if (distance < min_dist) {
              min_dist = distance;
              cluster_idx = j;
              best_iou = iou;
            }
        }

        float anchor_w = anchors_data.centers.vals[cluster_idx][0]; //centers->data.fl[cluster_idx * 2];
        float anchor_h = anchors_data.centers.vals[cluster_idx][1]; //centers->data.fl[cluster_idx * 2 + 1];
        if (best_iou > 1 || best_iou < 0) { // || box_w > width || box_h > height) {
            printf(" Wrong label: i = %d, box_w = %f, box_h = %f, anchor_w = %f, anchor_h = %f, iou = %f \n",
                i, box_w, box_h, anchor_w, anchor_h, best_iou);
        }
        else avg_iou += best_iou;
    }

    char buff[1024];
    FILE* fwc = fopen("counters_per_class.txt", "wb");
    if (fwc) {
        sprintf(buff, "counters_per_class = ");
        printf("\n%s", buff);
        fwrite(buff, sizeof(char), strlen(buff), fwc);
        for (i = 0; i < classes; ++i) {
            sprintf(buff, "%d", counter_per_class[i]);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fwc);
            if (i < classes - 1) {
                fwrite(", ", sizeof(char), 2, fwc);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fwc);
    }
    else {
        printf(" Error: file counters_per_class.txt can't be open \n");
    }

    avg_iou = 100 * avg_iou / number_of_boxes;
    printf("\n avg IoU = %2.2f %% \n", avg_iou);


    FILE* fw = fopen("anchors.txt", "wb");
    if (fw) {
        printf("\nSaving anchors to the file: anchors.txt \n");
        printf("anchors = ");
        for (i = 0; i < num_of_clusters; ++i) {
            float anchor_w = anchors_data.centers.vals[i][0]; //centers->data.fl[i * 2];
            float anchor_h = anchors_data.centers.vals[i][1]; //centers->data.fl[i * 2 + 1];
            if (width > 32) sprintf(buff, "%3.0f,%3.0f", anchor_w, anchor_h);
            else sprintf(buff, "%2.4f,%2.4f", anchor_w, anchor_h);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fw);
            if (i + 1 < num_of_clusters) {
                fwrite(", ", sizeof(char), 2, fw);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fw);
    }
    else {
        printf(" Error: file anchors.txt can't be open \n");
    }

    if (show) {
#ifdef OPENCV
        show_acnhors(number_of_boxes, num_of_clusters, rel_width_height_array, anchors_data, width, height);
#endif // OPENCV
    }
    free(rel_width_height_array);
    free(counter_per_class);
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    double time = what_time_is_it_now();

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
            error("fopen failed", DARKNET_LOC);
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;

        double load_time = (what_time_is_it_now() - time);
        printf("Loaded: %lf seconds", load_time);;

        if(letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        double resize_time = (what_time_is_it_now() - time);
        printf("Resize: %lf seconds", resize_time);;

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                l = lk;
                printf(" Detection layer: %d - type = %d \n", k, l.type);
            }
        }

        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

        float *X = sized.data;

        //time= what_time_is_it_now();
        double time = get_time_point();
        network_predict(net, X);
        //network_predict_image(&net, im); letterbox = 1;
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
        //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
        save_image(im, "predictions");
        if (!dont_show) {
            show_image(im, "predictions");
        }

        if (json_file) {
            if (json_buf) {
                char *tmp = ", \n";
                fwrite(tmp, sizeof(char), strlen(tmp), json_file);
            }
            ++json_image_id;
            json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, input);

            fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
            free(json_buf);
        }

        // pseudo labeling concept - fast.ai
        if (save_labels)
        {
            char labelpath[4096];
            replace_image_to_label(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
                char buff[1024];
                int class_id = -1;
                float prob = 0;
                for (j = 0; j < l.classes; ++j) {
                    if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
                        prob = dets[i].prob[j];
                        class_id = j;
                    }
                }
                if (class_id >= 0) {
                    sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            }
            fclose(fw);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }

    if (json_file) {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    free_network(net);
}

#if defined(OPENCV) && defined(GPU)

// adversarial attack dnn
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);// parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    net.adversarial = 1;
    set_batch_network(&net, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    //fuse_conv_batchnorm(net);
    //calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
    }

    srand(2222222);
    char buff[256];
    char *input = buff;

    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if (letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        image src_sized = copy_image(sized);

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                l = lk;
                printf(" Detection layer: %d - type = %d \n", k, l.type);
            }
        }

        net.num_boxes = l.max_boxes;
        int num_truth = l.truths;
        float *truth_cpu = (float *)xcalloc(num_truth, sizeof(float));

        int *it_num_set = (int *)xcalloc(1, sizeof(int));
        float *lr_set = (float *)xcalloc(1, sizeof(float));
        int *boxonly = (int *)xcalloc(1, sizeof(int));

        cv_draw_object(sized, truth_cpu, net.num_boxes, num_truth, it_num_set, lr_set, boxonly, l.classes, names);

        net.learning_rate = *lr_set;
        it_num = *it_num_set;

        float *X = sized.data;

        mat_cv* img = NULL;
        float max_img_loss = 5;
        int number_of_lines = 100;
        int img_size = 1000;
        char windows_name[100];
        char *base = basecfg(cfgfile);
        sprintf(windows_name, "chart_%s.png", base);
        img = draw_train_chart(windows_name, max_img_loss, it_num, number_of_lines, img_size, dont_show, NULL);

        int iteration;
        for (iteration = 0; iteration < it_num; ++iteration)
        {
            forward_backward_network_gpu(net, X, truth_cpu);

            float avg_loss = get_network_cost(net);
            draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, it_num, 0, 0, "mAP%", 0, dont_show, 0, 0);

            float inv_loss = 1.0 / max_val_cmp(0.01, avg_loss);
            //net.learning_rate = *lr_set * inv_loss;

            if (*boxonly) {
                int dw = truth_cpu[2] * sized.w, dh = truth_cpu[3] * sized.h;
                int dx = truth_cpu[0] * sized.w - dw / 2, dy = truth_cpu[1] * sized.h - dh / 2;
                image crop = crop_image(sized, dx, dy, dw, dh);
                copy_image_inplace(src_sized, sized);
                embed_image(crop, sized, dx, dy);
            }

            show_image_cv(sized, "image_optimization");
            wait_key_cv(20);
        }

        net.train = 0;
        quantize_image(sized);
        network_predict(net, X);

        save_image_png(sized, "drawn");
        //sized = load_image("drawn.png", 0, 0, net.c);

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, 0, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(sized, dets, nboxes, thresh, names, alphabet, l.classes, 1);
        save_image(sized, "pre_predictions");
        if (!dont_show) {
            show_image(sized, "pre_predictions");
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        free_image(src_sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        free(lr_set);
        free(it_num_set);

        if (filename) break;
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}

/*
 * Resolve a file's physical block ranges with FS_IOC_FIEMAP.
 *
 * The target has no filesystem view of the device, so the host must hand it
 * raw LBA ranges. This requires the file to be on the NVMe-oF volume and
 * already flushed to disk; a file still sitting in the page cache yields
 * extents that do not describe its on-disk location.
 */
static int ndp_get_extents(const char *filepath, ndp_extent_table_t *out)
{
    /* Heap-allocated: putting the trailing extent array on the stack would be
     * a variable-length array of unbounded size. */
    size_t fm_sz = sizeof(struct fiemap)
                 + NDP_MAX_EXTENTS * sizeof(struct fiemap_extent);
    struct fiemap *fm = (struct fiemap *)calloc(1, fm_sz);
    if (!fm) {
        fprintf(stderr, "[NDP] calloc for fiemap failed\n");
        return -1;
    }
 
    fm->fm_length       = FIEMAP_MAX_OFFSET;
    fm->fm_extent_count = NDP_MAX_EXTENTS;
 
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("[NDP] open (fiemap)");
        free(fm);
        return -1;
    }
    if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) {
        perror("[NDP] FS_IOC_FIEMAP");
        close(fd);
        free(fm);
        return -1;
    }
    close(fd);
 
    uint32_t n = fm->fm_mapped_extents;
    if (n == 0) {
        fprintf(stderr, "[NDP] No extents found for: %s\n", filepath);
        free(fm);
        return -1;
    }
    if (n > NDP_MAX_EXTENTS) {
        fprintf(stderr, "[NDP] Warning: extent count %u > max %d, truncating. "
                        "The tail of the file will not be processed.\n",
                n, NDP_MAX_EXTENTS);
        n = NDP_MAX_EXTENTS;
    }
 
    for (uint32_t i = 0; i < n; i++) {
        out->lba[i] = fm->fm_extents[i].fe_physical / NDP_BLOCK_SIZE;
        out->blk[i] = fm->fm_extents[i].fe_length   / NDP_BLOCK_SIZE;
    }
    out->count = n;
 
    free(fm);
    return 0;
}
 
/*
 * 0xC0 - trigger preprocessing on the storage node.
 *
 * Payload : serialised extent table, [lba0, blk0, lba1, blk1, ...]
 * cdw10   : extent count
 * cdw11   : sample rate, keep 1 frame out of N (0 means 1)
 * cdw12   : [7:0] scaler selector, [15:8] JPEG quality (0 means target default)
 * cdw0    : (response) total byte size of the result buffer
 *
 * The ioctl blocks until the target's worker thread has finished the entire
 * clip, so no polling loop is needed on the host side.
 *
 * Result buffer layout produced by the target:
 *   [uint32 frame_count][uint32 jpeg_size x N][JPEG blob x N]
 * Frame sizes vary, hence the explicit size table instead of a fixed stride.
 */
static int ndp_send_0xc0(int nvme_fd, const char *mp4_path,
                         int sample_rate, int scaler_sel, int jpeg_quality,
                         uint32_t *total_result_size_out)
{
    ndp_extent_table_t ext = {0};
    if (ndp_get_extents(mp4_path, &ext) < 0)
        return -1;
 
    printf("[NDP] 0xC0: %u extents for %s\n", ext.count, mp4_path);
    for (uint32_t i = 0; i < ext.count; i++)
        printf("      [%u] LBA=%-12llu blocks=%llu\n", i,
               (unsigned long long)ext.lba[i],
               (unsigned long long)ext.blk[i]);
 
    size_t tbl_bytes = (size_t)ext.count * 2 * sizeof(uint64_t);
    size_t buf_bytes = (tbl_bytes + 4095) & ~(size_t)4095;   /* DMA alignment */
 
    void *tbl = aligned_alloc(4096, buf_bytes);
    if (!tbl) {
        fprintf(stderr, "[NDP] aligned_alloc failed (%zu bytes)\n", buf_bytes);
        return -1;
    }
    memset(tbl, 0, buf_bytes);
    uint64_t *u64 = (uint64_t *)tbl;
    for (uint32_t i = 0; i < ext.count; i++) {
        u64[2 * i]     = ext.lba[i];
        u64[2 * i + 1] = ext.blk[i];
    }
 
    uint32_t cdw12 = ((uint32_t)(scaler_sel   & 0xFF))
                   | ((uint32_t)(jpeg_quality & 0xFF) << 8);
 
    struct nvme_passthru_cmd cmd = {
        .opcode     = 0xC0,
        .flags      = 0,
        .nsid       = 1,
        .addr       = (uint64_t)(uintptr_t)tbl,
        .data_len   = (uint32_t)buf_bytes,
        .cdw10      = ext.count,
        .cdw11      = (uint32_t)(sample_rate & 0xFFFF),
        .cdw12      = cdw12,
        .timeout_ms = NDP_NVME_TIMEOUT_MS,
    };
 
    printf("[NDP] Sending 0xC0 (data_len=%zu, cdw10=%u, sample_rate=1/%d, "
           "scaler_sel=%d, jpeg_q=%d, cdw12=0x%08x) ...\n",
           buf_bytes, ext.count, sample_rate, scaler_sel,
           jpeg_quality ? jpeg_quality : 85, cdw12);
    fflush(stdout);
 
    int ret = ioctl(nvme_fd, NVME_IOCTL_IO_CMD, &cmd);
    free(tbl);
 
    if (ret != 0) {
        fprintf(stderr, "[NDP] 0xC0 ioctl error: ret=%d errno=%d (%s)\n",
                ret, errno, strerror(errno));
        return -1;
    }
 
    uint32_t total = cmd.result;
    printf("[NDP] 0xC0 done: cdw0=%u bytes\n", total);
 
    if (total_result_size_out) *total_result_size_out = total;
    return 0;
}
 
/*
 * 0xC2 - retrieve the result buffer.
 *
 * Issued repeatedly with cdw10 carrying the current byte offset. The chunk
 * size is bounded by the controller's maximum data transfer size (MDTS); the
 * target frees the buffer and returns to idle after serving the final chunk.
 */
static int ndp_send_0xc2(int nvme_fd, void *buf, uint32_t total_bytes)
{
    const uint32_t mdts_limit = 2 * 1024 * 1024;
    uint32_t offset = 0;
 
    while (offset < total_bytes) {
        uint32_t current_len = (total_bytes - offset > mdts_limit)
                             ? mdts_limit : (total_bytes - offset);
 
        struct nvme_passthru_cmd cmd = {
            .opcode     = 0xC2,
            .nsid       = 1,
            .addr       = (uint64_t)(uintptr_t)((uint8_t *)buf + offset),
            .data_len   = current_len,
            .cdw10      = offset,
            .timeout_ms = NDP_NVME_TIMEOUT_MS,
        };
 
        int ret = ioctl(nvme_fd, NVME_IOCTL_IO_CMD, &cmd);
        if (ret != 0) {
            fprintf(stderr, "[NDP] 0xC2 ioctl error: ret=%d errno=%d (%s)\n",
                    ret, errno, strerror(errno));
            return -1;
        }
 
        offset += current_len;
    }
    return 0;
}
 
/* ══════════════════════════════════════════════════════════════
 * 공개 함수: ndp_test_detector
 *
 * detector.c의 run_detector()에서 "ndp_test" 서브커맨드로 호출.
 * 헤더 선언은 demo.h 또는 별도 ndp_detector.h에 추가 필요:
 *   void ndp_test_detector(char*, char*, char*, char*, char*,
 *                          float, float, int, int, char*, int, int);
 * ══════════════════════════════════════════════════════════════ */
 
void ndp_test_detector(char *datacfg, char *cfgfile, char *weightfile,
                       char *nvme_dev, char *mp4_path,
                       float thresh, float hier_thresh,
                       int dont_show, int ext_output, char *outfile,
                       int letter_box, int benchmark_layers,
                       int ndp_sample_rate)
{
    /* ── 1. darknet 네트워크 로딩 ─────────────────────────── */
    list *options    = read_data_cfg(datacfg);
    char *name_list  = option_find_str(options, "names", "data/names.list");
    int   names_size = 0;
    char **names     = get_labels_custom(name_list, &names_size);
    image **alphabet = load_alphabet();

    double time = what_time_is_it_now();
 
    network net = parse_network_cfg_custom(cfgfile, 1, 1);  /* batch=1 */
    if (weightfile) load_weights(&net, weightfile);
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);
 
    /* 마지막 detection layer 찾기 */
    layer l = net.layers[net.n - 1];
    for (int k = 0; k < net.n; k++) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION)
            l = lk;
    }
 
    if (l.classes != names_size) {
        fprintf(stderr, "[NDP] Warning: cfg classes=%d but names=%d\n",
                l.classes, names_size);
    }
 
    /* ── 2. NVMe 디바이스 열기 ───────────────────────────── */
    int nvme_fd = open(nvme_dev, O_RDWR);
    if (nvme_fd < 0) {
        fprintf(stderr, "[NDP] Cannot open NVMe device %s: %s\n",
                nvme_dev, strerror(errno));
        goto cleanup_net;
    }
    printf("[NDP] Opened NVMe device: %s\n", nvme_dev);
 
    /* ── 3. 0xC0: NDP 디코딩 + 샘플링 트리거 ────────────── */
    /*
     * ioctl은 타겟의 ndp_preprocess_worker() 완료 시점까지 블록킹.
     * 타겟: SPDK_NVMF_REQUEST_EXEC_STATUS_ASYNCHRONOUS 리턴 후
     *       spdk_nvmf_request_complete() 호출 → CQE 전송 → ioctl 리턴.
     * cdw0(result) = full_video_buffer 전체 바이트 크기
     *   ([uint32 frame_count][uint32 jpeg_size × N][JPEG blob × N])
     */
    uint32_t total_result_size = 0;
      if (ndp_send_0xc0(nvme_fd, mp4_path, ndp_sample_rate, 0, 0, &total_result_size) < 0)
        goto cleanup_fd;
 
    if (total_result_size == 0) {
        fprintf(stderr, "[NDP] Target returned 0 bytes. "
                        "Check g_total_frames on target side.\n");
        goto cleanup_fd;
    }

    double ndp_decoding_time = (what_time_is_it_now() - time);
        printf("ndp_decoding_time: %lf seconds\n", ndp_decoding_time);
 
    /* ── 4. 0xC2: 전처리 결과 버퍼 전체 fetch ───────────── */
    size_t alloc_bytes  = ((size_t)total_result_size + 4095) & ~(size_t)4095;
    uint8_t *result_buf = (uint8_t *)aligned_alloc(4096, alloc_bytes);
    fprintf(stdout, "[NDP] alloc %zu bytes\n", alloc_bytes);
    if (!result_buf) {
        fprintf(stderr, "[NDP] Failed to alloc %zu bytes\n", alloc_bytes);
        goto cleanup_fd;
    }
    memset(result_buf, 0, alloc_bytes);

    if (ndp_send_0xc2(nvme_fd, result_buf, total_result_size) < 0)
        goto cleanup_buf;

    double get_result_time = (what_time_is_it_now() - time);
        printf("get_result_time: %lf seconds\n", get_result_time);

    /* ── 4-1. 결과 헤더 파싱 ─────────────────────────────
     *   [uint32 frame_count][uint32 jpeg_size × N][JPEG blob × N]
     */
    uint32_t num_frames = 0;
    memcpy(&num_frames, result_buf, sizeof(uint32_t));

    size_t header_bytes = (size_t)(1 + num_frames) * sizeof(uint32_t);
    if (num_frames == 0 || header_bytes > (size_t)total_result_size) {
        fprintf(stderr, "[NDP] Invalid result header: frame_count=%u, "
                        "total=%u bytes\n", num_frames, total_result_size);
        goto cleanup_buf;
    }

    const uint32_t *jpeg_sizes = (const uint32_t *)(result_buf + sizeof(uint32_t));
    const uint8_t  *blob_base  = result_buf + header_bytes;

    size_t blob_total = 0;
    for (uint32_t i = 0; i < num_frames; i++) blob_total += jpeg_sizes[i];
    if (header_bytes + blob_total > (size_t)total_result_size) {
        fprintf(stderr, "[NDP] Result size mismatch: header %zu + blobs %zu "
                        "> total %u\n", header_bytes, blob_total, total_result_size);
        goto cleanup_buf;
    }

    printf("[NDP] Frames: %u, JPEG payload: %zu bytes "
           "(avg %.1f KB/frame)\n",
           num_frames, blob_total, blob_total / 1024.0 / num_frames);
 
    /* ── 5. JSON 출력 파일 준비 ──────────────────────────── */
    FILE *json_file = NULL;
    char *json_buf  = NULL;
    int   json_id   = 0;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if (!json_file) {
            fprintf(stderr, "[NDP] Cannot open outfile: %s\n", outfile);
        } else {
            fwrite("[\n", 1, 2, json_file);
        }
    }
 
    /* ── 6. 프레임별 추론 루프 ───────────────────────────── */
    const float nms = 0.45f;

    size_t blob_off      = 0;
    double t_jpegdec_sum = 0.0;

    for (uint32_t fi = 0; fi < num_frames; fi++) {
        /* storage가 보낸 JPEG(net 크기 letterbox 완료) → darknet image.
         * 복호화·색변환·letterbox는 이미 storage에서 끝났으므로
         * host는 JPEG 압축 해제만 수행한다. */
        double t_dec = what_time_is_it_now();
        image im_input = ndp_jpeg_to_image(blob_base + blob_off, jpeg_sizes[fi]);
        t_jpegdec_sum += (what_time_is_it_now() - t_dec);
        blob_off += jpeg_sizes[fi];

        if (!im_input.data || im_input.w == 0 || im_input.h == 0) {
            fprintf(stderr, "[NDP] Frame %u: JPEG decode failed, skipped\n", fi);
            if (im_input.data) free_image(im_input);
            continue;
        }
        if (im_input.w != net.w || im_input.h != net.h) {
            fprintf(stderr, "[NDP] Frame %u: size %dx%d != net %dx%d\n",
                    fi, im_input.w, im_input.h, net.w, net.h);
            free_image(im_input);
            continue;
        }
 
        /* 추론 */
        double t0 = get_time_point();
        network_predict(net, im_input.data);
        printf("[NDP] Frame %u/%u: predicted in %.1f ms\n",
               fi + 1, num_frames,
               (get_time_point() - t0) / 1000.0);
 
        /* 검출 결과 */
        int nboxes = 0;
        detection *dets = get_network_boxes(
            &net, im_input.w, im_input.h,
            thresh, hier_thresh, 0, 1, &nboxes, letter_box);
 
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS)
                do_nms_sort(dets, nboxes, l.classes, nms);
            else
                diounms_sort(dets, nboxes, l.classes, nms,
                             l.nms_kind, l.beta_nms);
        }
 
        /* 바운딩 박스 오버레이 */
        draw_detections_v3(im_input, dets, nboxes, thresh,
                           names, alphabet, l.classes, ext_output);
 
        /* 결과 이미지 저장 (ndp_frame_0000.jpg, ...) */
        char save_name[64];
        snprintf(save_name, sizeof(save_name), "ndp_frame_%04u", fi);
        save_image(im_input, save_name);
        printf("[NDP] Saved: %s.jpg (%d detections)\n", save_name, nboxes);
 
        /* 첫 프레임만 화면 표시 (dont_show 옵션 아닌 경우) */
        if (!dont_show && fi == 0) {
            show_image(im_input, "NDP Detection - first sampled frame");
        }
 
        /* JSON 기록 */
        if (json_file) {
            if (json_buf) fwrite(", \n", 1, 3, json_file);
            ++json_id;
            json_buf = detection_to_json(dets, nboxes, l.classes,
                                         names, json_id, save_name);
            fwrite(json_buf, 1, strlen(json_buf), json_file);
            free(json_buf);
            json_buf = NULL;
        }
 
        free_detections(dets, nboxes);
        free_image(im_input);
    }

    printf("[NDP] JPEG decode total: %lf seconds (%.2f ms/frame)\n",
           t_jpegdec_sum, t_jpegdec_sum * 1000.0 / num_frames);
 
    /* ── 7. 정리 ─────────────────────────────────────────── */
    if (json_file) {
        fwrite("\n]", 1, 2, json_file);
        fclose(json_file);
        printf("[NDP] JSON results written to: %s\n", outfile);
    }
    if (!dont_show) {
        wait_until_press_key_cv();
        destroy_all_windows_cv();
    }
    printf("[NDP] Done. Processed %u/%u sampled frames from %s\n",
           num_frames, num_frames, mp4_path);
 
cleanup_buf:
    free(result_buf);
cleanup_fd:
    close(nvme_fd);
cleanup_net:
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);
    free_alphabet(alphabet);
    free_network(net);

    double total_time = (what_time_is_it_now() - time);
            printf("total_time: %lf seconds \n", total_time);   
}


#else // defined(OPENCV) && defined(GPU)
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    error("darknet detector draw ... can't be used without OpenCV and CUDA", DARKNET_LOC);
}
#endif // defined(OPENCV) && defined(GPU)

static void ndp_print_usage(const char *prog)
{
    printf(
"NDP subcommands (NVMe-oF near-data video preprocessing)\n"
"\n"
"  %s detector ndp_test  <data> <cfg> <weights> [options]\n"
"      Offload preprocessing to the storage node and run detection on the\n"
"      returned frames. Writes an annotated JPEG per frame.\n"
"\n"
"  %s detector video_map <data> <cfg> <weights> -file <video> -labels <dir> [options]\n"
"      Measure detection accuracy (mAP) over a video, either through the host\n"
"      baseline or through the NDP pipeline.\n"
"\n"
"Common options\n"
"  -file <path>        Input video. With -ndp the path must be on the\n"
"                      NVMe-oF volume, and the file must be flushed (sync).\n"
"  -dev <path>         NVMe character device for passthrough commands.\n"
"                      Default /dev/nvme1n1. Requires read/write permission.\n"
"  -sample_rate <N>    Keep one frame out of every N. Sent to the target in\n"
"                      cdw11. ndp_test defaults to 5, video_map to 1.\n"
"  -dont_show          Do not open any display window.\n"
"\n"
"video_map options\n"
"  -labels <dir>       Ground-truth directory holding <frame>.txt files in\n"
"                      YOLO format (class cx cy w h, normalised to the source\n"
"                      resolution). Required.\n"
"  -label_base <N>     Label file number for the first decoded frame.\n"
"                      Default 1 (MOT-style datasets start at 000001).\n"
"  -ndp                Evaluate the NDP pipeline. Without this flag the host\n"
"                      baseline is evaluated instead.\n"
"  -iou_thresh <f>     IoU threshold for the primary AP figure. Default 0.5.\n"
"  -points <N>         Recall points for AP integration: 101 for COCO,\n"
"                      11 for VOC 2007, 0 for area under the curve.\n"
"  -coco_map           Additionally report mAP averaged over\n"
"                      IoU = 0.50:0.05:0.95.\n"
"  -thresh <f>         Confidence threshold for the precision/recall/F1\n"
"                      summary only. Default 0.25. AP always integrates from\n"
"                      0.005 and is unaffected.\n"
"\n"
"Ablation options (NDP only, forwarded to the target in cdw12)\n"
"  -scaler <N>         libswscale kernel used for downscaling on the target.\n"
"                        0 BILINEAR (default)   1 POINT\n"
"                        2 FAST_BILINEAR        3 BICUBIC\n"
"                        4 AREA                 5 LANCZOS\n"
"  -jpeg_q <N>         JPEG quality for re-encoding, 1-100. 0 uses the\n"
"                      target default of 85.\n"
"\n"
"Examples\n"
"  # Host baseline, every frame, COCO-style mAP\n"
"  %s detector video_map mot.data yolov7-tiny.cfg yolov7-tiny.weights \\\n"
"      -file clip.mp4 -labels labels/clip -points 101 -coco_map -dont_show\n"
"\n"
"  # NDP pipeline, 1 frame in 5\n"
"  %s detector video_map mot.data yolov7-tiny.cfg yolov7-tiny.weights \\\n"
"      -ndp -dev /dev/nvme1n1 -file /mnt/ext4/clip.mp4 -labels labels/clip \\\n"
"      -sample_rate 5 -points 101 -coco_map -dont_show\n"
"\n"
"  # Ablation: nearest-neighbour scaling, lossless JPEG\n"
"  %s detector video_map mot.data yolov7-tiny.cfg yolov7-tiny.weights \\\n"
"      -ndp -file /mnt/ext4/clip.mp4 -labels labels/clip \\\n"
"      -sample_rate 5 -scaler 1 -jpeg_q 100 -points 101 -coco_map -dont_show\n"
"\n", prog, prog, prog, prog, prog);
}

void run_detector(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = 1;
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
    int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
    int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *json_file_output = find_char_arg(argc, argv, "-json_file_output", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);
    // While training, decide after how many epochs mAP will be calculated. Default value is 4 which means the mAP will be calculated after each 4 epochs
    int mAP_epochs = find_int_arg(argc, argv, "-mAP_epochs", 4);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if (0 == strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    else if (0 == strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path, mAP_epochs);
    else if (0 == strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if (0 == strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
    else if (0 == strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL);
    else if (0 == strcmp(argv[2], "calc_anchors")) calc_anchors(datacfg, num_of_clusters, width, height, show);
    else if (0 == strcmp(argv[2], "draw")) {
        int it_num = 100;
        draw_object(datacfg, cfg, weights, filename, thresh, dont_show, it_num, letter_box, benchmark_layers);
    }
    else if (0 == strcmp(argv[2], "ndp_help") ||
             0 == strcmp(argv[2], "ndp_usage")) {
        ndp_print_usage(argv[0]);
    }
    else if (0 == strcmp(argv[2], "ndp_test")) {
        char *nvme_dev = find_char_arg(argc, argv, "-dev", "/dev/nvme1n1");
        char *mp4_path = find_char_arg(argc, argv, "-file", NULL);
        int   ndp_sample_rate = find_int_arg(argc, argv, "-sample_rate", 5);
        if (!mp4_path || find_arg(argc, argv, "-help")) {
            ndp_print_usage(argv[0]);
            return;
        }
        ndp_test_detector(datacfg, cfg, weights, nvme_dev, mp4_path,
                          thresh, hier_thresh, dont_show, ext_output,
                          outfile, letter_box, benchmark_layers,
                          ndp_sample_rate);
    }
    else if (0 == strcmp(argv[2], "video_map")) {
        char *nvme_dev    = find_char_arg(argc, argv, "-dev",    "/dev/nvme1n1");
        char *video_path  = find_char_arg(argc, argv, "-file",   NULL);
        char *label_dir   = find_char_arg(argc, argv, "-labels", NULL);
        int   use_ndp     = find_arg(argc, argv, "-ndp");
        int   sample_rate = find_int_arg(argc, argv, "-sample_rate", 1);
        int   label_base  = find_int_arg(argc, argv, "-label_base", 1);
        int   coco_range  = find_arg(argc, argv, "-coco_map");
        /* 0 means "use whatever the target defaults to" */
        int   scaler_sel  = find_int_arg(argc, argv, "-scaler", 0);
        int   jpeg_q      = find_int_arg(argc, argv, "-jpeg_q", 0);
        if (!video_path || !label_dir || find_arg(argc, argv, "-help")) {
            ndp_print_usage(argv[0]);
            return;
        }
        validate_video_map(datacfg, cfg, weights, video_path, label_dir,
                           nvme_dev, use_ndp, sample_rate, label_base,
                           thresh, iou_thresh, map_points, coco_range,
                           scaler_sel, jpeg_q);
    }
    else if (0 == strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        if (filename)
            if (strlen(filename) > 0)
                if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, avgframes, frame_skip, prefix, out_filename,
            mjpeg_port, dontdraw_bbox, json_port, dont_show, ext_output, letter_box, time_limit_sec, http_post_host, benchmark, benchmark_layers, json_file_output);

        free_list_contents_kvp(options);
        free_list(options);
    }
    else printf(" There isn't such command: %s", argv[2]);

    if (gpus && gpu_list && ngpus > 1) free(gpus);
}