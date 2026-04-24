#!/bin/bash

# 1. 설정
DARKNET_PATH="."
VIDEO_DIR="/mnt/ext4"
# 결과 저장 메인 디렉터리 (날짜/시간 포함)
RESULT_ROOT="ndp_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_ROOT"

# 실험할 파일 리스트 가져오기 (testb로 시작하고 fps가 포함된 mp4 파일들)
FILES=$(ls $VIDEO_DIR/testb_*fps_std.mp4)

echo "================================================================"
echo "NDP Batch Test Start: $(echo $FILES | wc -w) files found."
echo "Results will be saved in: $RESULT_ROOT"
echo "================================================================"

for FILE_PATH in $FILES; do
    # 파일명만 추출 (예: testb-1080_30fps)
    FILE_NAME=$(basename "$FILE_PATH" .mp4)
    
    # 각 영상별 결과 디렉터리 생성
    OUTPUT_DIR="$RESULT_ROOT/$FILE_NAME"
    mkdir -p "$OUTPUT_DIR"
    
    echo "[Testing] $FILE_NAME ..."

    # 2. Darknet 실행
    # stdout과 stderr을 모두 log.txt에 저장하면서 동시에 화면에도 출력(tee)
    sudo ./darknet detector ndp_test data/coco.data cfg/yolov7-tiny.cfg yolov7-tiny.weights \
        -dev /dev/nvme1n1 \
        -file "$FILE_PATH" \
        -thresh 0.25 \
        -dont_show \
        -letter_box \
        -out "$OUTPUT_DIR/${FILE_NAME}_result.json" 2>&1 | tee "$OUTPUT_DIR/log.txt"

    # 3. 생성된 이미지 파일들(.jpg)을 해당 디렉터리로 이동
    # Darknet이 현재 디렉터리에 ndp_frame_0000.jpg 등으로 생성하므로 이를 이동시킴
    if ls ndp_frame_*.jpg 1> /dev/null 2>&1; then
        mv ndp_frame_*.jpg "$OUTPUT_DIR/"
    fi

    echo "[Done] Results saved to $OUTPUT_DIR"
    echo "----------------------------------------------------------------"
done

echo "All NDP tests completed."
