# 연산 구조별 모델 검증 후보

이 문서는 조사와 시험 우선순위다. 목록 등록, 파일 다운로드, 컴파일 성공, NPU 실행 검증은 각각 별개의 상태다. 모델이 여러 구성요소로 나뉘는 경우 모델 전체와 개별 그래프의 결과를 분리해 기록한다. 크기는 가중치 파일 용량이며 실행 시 메모리가 아니다.

| 순서 | 대상 | 확인할 구조와 경계 | 현황 |
|---|---|---|---|
| 1 | `Ultralytics/YOLO11`, `Ultralytics/YOLO26` | Conv, resize, concat, 탐지 헤드, 전·후처리. YOLO26은 기본 NMS와 `nms=False` 두 경로 | YOLO11 ONNX 별도 경로의 이전 NPU 증거 존재. 캐시된 Ultralytics `.pt`와 동일한 검증으로 취급하지 않음. YOLO26 후보 등록, 미다운로드·미검증 |
| 2 | `nvidia/segformer-b0-finetuned-ade-512-512`, `apple/deeplabv3-mobilevit-xx-small` | 공간 크기 변화, 마스크, 보간·업샘플링 | 기존 카탈로그 후보, NPU 미검증 |
| 3 | `openai/whisper-tiny`, `openai/whisper-small` | Conv1d, 인코더-디코더 교차 attention, 디코딩 반복 | 기존 카탈로그 후보. tiny 다운로드 이전에 실패한 이력 존재 |
| 4 | `facebook/dinov3-vits16-pretrain-lvd1689m` | ViT patch·RoPE·입력 해상도별 그래프 | 공식 모델 페이지에 접근 동의 필요 표시. 다운로드 후보로 자동 분류하지 않음 |
| 5 | `state-spaces/mamba2-130m`, `state-spaces/mamba2-370m` | 상태 업데이트, selective scan, 시퀀스 길이별 경로 | 후보 등록, 모델 코드·토크나이저 의존성 및 NPU 미검증 |
| 6 | `facebook/sam3` | 텍스트·이미지 조건, 마스크, 복합 모델 경계 | 공식 모델 페이지에 접근 동의 필요 표시. 다운로드 후보로 자동 분류하지 않음 |

출처: [YOLO26 설명](https://docs.ultralytics.com/models/yolo26/), [YOLO26n 파일](https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n.pt), [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper), [DINOv3](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), [Mamba-2 논문](https://arxiv.org/abs/2405.21060), [SAM3](https://huggingface.co/facebook/sam3).

과거 YOLO11 보고서 `/root/yolo11-rngd/PORTING_REPORT.json`은 `giangndm/yolo11-onnx` revision `8b180c762d5cf217886d16e64403508042207e4d`에 대한 640×640 FP32 시험을 기록한다. 보고된 런타임 추적은 NPU 88/88 그래프, CPU fallback 없음이며 전처리와 NMS는 호스트 작업이다. 이것은 다른 체크포인트나 새로운 재실행에 대한 근거가 아니다.
