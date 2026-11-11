# 정적 FXB 실패 수집

TCLMachine의 `export_model`이 만드는 정적 EDF·실행 계획과 fxbX의 적재/실행을 검증한다. 기존 `optimize`/direct-runtime 결과나 공식 Furiosa-LLM FXB 결과와 섞지 않는다. 자동 수정·SDK 교체·모델 가중치 다운로드를 하지 않는다.

## 사용

검증 환경은 5번 서버(`ssh furiosa-npu`), Python `/root/TCLMachine_test/.venv/bin/python`, TCLMachine `66251c4`, fxbX `f4025ef` 소스와 SDK 2026.4.0이다. 실행 전 `furiosa-smi ps`를 확인하고 비어 있는 장치를 사용한다.

설치된 hf_librarian의 Python 의존성이 SDK venv에 없다면 아래 독립 파일 실행을 사용한다. SDK 환경에 의존성을 새로 설치할 필요가 없다.

```bash
ssh furiosa-npu
export PYTHONPATH=/root/TCLMachine
PY=/root/TCLMachine_test/.venv/bin/python
TOOL=/root/hf_librarian_for_furiosa/hf_librarian/static_failures.py
$PY "$TOOL" run --out /root/.cache/hf-librarian/static-failures/my-new-attempt --consumer-source /root/fxbX
```

hf_librarian이 설치된 같은 환경에서는 `python -m hf_librarian.static_failures`도 동일하다. `--cases minilm whisper segformer`로 범위를 줄이고 `--timeout 900`으로 **모델당** 제한 시간을 지정한다. 순차 실행하며 컴파일 worker는 2개다. HF 다운로드 동시성 설정과 별개다. 출력 디렉터리는 매번 새 이름을 사용한다. 과거 시도/실패를 덮어쓰지 않는다. timeout 시 해당 worker의 프로세스 그룹만 종료하며 다른 사용자의 작업을 종료하지 않는다.

캠페인의 소비자를 고정하려면 `--consumer-source /path/to/fxbX`를 추가한다. 이 경로에는 `fxbx/`와 `fxbex/`가 함께 있어야 한다. 작업자와 그 자식 프로세스에서 해당 소스를 검색 경로의 앞에 놓고, 작업자가 실제로 불러온 두 모듈의 경로를 검사한다. 옵션을 생략하면 SDK 환경에서 찾은 소비자를 사용한다.

`result.json`은 실제 소비자 모듈 경로와 SHA256, API 버전, 설치 배포판 버전, frozen-buffer 검증 지원 여부를 기록한다. 서로 다른 소스가 같은 버전을 표시할 수 있다. `consumer_capability_deferrals`가 있으면 필요한 기능을 갖춘 소비자로 같은 모델 리비전과 입력을 사전 검사한다. 66251c4 재검증에서 설치된 `0.8.3`은 frozen-buffer 검증을 지원하지 않았고, 같은 버전을 표시하는 fxbX main `7b34144`의 실행 코드에는 `f4025ef`의 검증이 포함돼 있다.

같은 컴파일 요청을 재시도할 때는 `--artifact-root /path/to/existing/edfs`로 기존 EDF 캐시를 지정할 수 있다. 컴파일러의 캐시 키와 무결성 검사가 재사용 여부를 결정한다. 옵션을 생략하면 각 시도의 `edfs/`를 사용한다. 실제 경로는 `command.json`과 `result.json`에 기록한다.

MiniLM `[1,16]` 재시험에서 `--consumer-source`와 `--artifact-root`를 함께 사용해 EDF 33개를 모두 재사용했다. FXB 생성과 수치·NPU 확인은 12.92초에 완료됐다. 실행 시 비영구 버퍼도 이름과 레이아웃을 보존한 채 로더에 전달한다.

## 대표성

|case|서로 다른 확인 대상|입력 범위|
|---|---|---|
|control|선형·ReLU 대조군|4×8 BF16|
|minilm|BERT encoder, LayerNorm, embedding|16 token int32 ID + mask|
|deit|시각 patch/position embedding·Attention|224×224 BF16; 실제 클래스/config도 기록|
|resnet|Conv2d·BatchNorm·잔차·pooling|224×224 BF16|
|segformer|다중 해상도 분할·depthwise conv·resize|64×64 BF16 smoke geometry|
|whisper|Conv1d·음성 encoder·cross Attention|고정 길이 zero mel + decoder 1 token|
|lfm2|short convolution과 Attention 혼합|16 token prefill|
|qwen35|gated delta/recurrent와 Attention 혼합|16 token text-only prefill|

체크포인트는 `CASES`에 명시한다. 크기만 다른 모델을 추가하는 대신 새로운 구조를 추가한다. 새 모델은 `CASES`와 `fixture()`의 입력 어댑터를 함께 추가한다. 컴파일/실행 경로에는 모델명 분기를 추가하지 않는다. 원본 모델 config와 가중치를 보존하고 BF16 시험 dtype을 명시한다. 입력은 결정적 합성 smoke 데이터이므로 실제 이미지/음성/언어 품질 평가가 아니다. LFM2/Qwen의 decode/KV 상태·긴 context, segmentation 전체 해상도는 별도 시험이다.

## 자동 기록

`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`를 사용한다. 캐시의 유일한 immutable snapshot 또는 기존 main ref만 사용하며 여러 revision 중 최신을 임의로 고르지 않는다. 캐시가 없거나 gated 모델이 미준비면 fixture 실패로 남긴다. 이번 구현은 HF 메타데이터 API도 호출하지 않는다.

각 모델 디렉터리:

- `result.json`: 상태, 단계별 경과 시각, model/revision/config hash, 입력 shape/dtype/stride/hash, 패키지·컴파일러 버전, 실행 호스트, 소스 hash, 실패 signature, traceback frame, 진단/해결 제안/다음 증거.
- `graph.json`: 캡처된 **전체 노드**의 연산·shape·dtype·stride·모듈 경로·소스 위치.
- `model.fxb.plan/preflight.json`: binding 실패 또는 규칙 부재의 그룹·발생 횟수·대표 위치. 생산자 API가 그룹을 최대 32개, 최초 실패를 8개로 제한하므로 전체 그래프와 함께 읽는다.
- `traceback.txt`, `process.log`, `command.json`, `process.json`: 원문 예외, stdout/stderr, argv/필수 환경과 실제 worker 종료코드. Python 프로세스 exit=0이어도 모델 `status=failed`일 수 있다.
- `sources.tar.gz`: 해당 시도에서 사용한 수집기·생산자·소비자 Python 소스, SHA256. 자동 복원/설치는 하지 않는다.
- 성공 시 `export.json`, 실제 `model.fxb`, `edfs/`, `trace.json`: 패키지와 컴파일 결과 및 공식 runtime trace.

캠페인 루트의 `summary.json`, `REPORT.md`, `issues.json`은 매 모델 종료 후 갱신한다. 결과를 다시 모을 때:

```bash
$PY "$TOOL" report --out /root/.cache/hf-librarian/static-failures/my-new-attempt
```

`fixture → capture → export_static → cpu_reference → reload → execute → numerical → trace_gate`를 구분한다. `export_static` 안에서는 TCLMachine preflight/code/로그와 성공 시 compile/packaging 시간을 이용해 세부 경계를 확인한다. exception이 나면 이후 단계는 실행되지 않는다. 실행 전 실패의 `NPU=not_run`은 NPU 자체 실패를 뜻하지 않는다. trace가 없으면 CPU fallback 유무는 `unknown`이다.

`EXECUTION_RULE_GAP`은 공유 EDF 규칙 부족이고 SDK/NPU의 원천적 미지원 판정이 아니다. `LAYOUT_UNSUPPORTED`는 tensor binding layout guard의 거부다. 확인된 guard와 미확인 근본 원인을 구분한다. 해결 제안은 아직 검증한 수정이 아니다. 첫 실패를 고친 뒤 같은 revision/입력/SDK로 새 시도를 만들면 뒤에 가려진 실패를 추가 수집할 수 있다.

NPU 판정은 모든 `execution_plan::` step의 공식 `Task` 대응, trace의 `run_by_cpu=0`, CPU 수치 비교를 요구한다. 단일 작은 smoke 성공은 모델 지원 인증이 아니다. changed input, 반복 실행, 상태/reset, **새 프로세스**의 컴파일러 없는 재적재는 후속 수락 조건이다. 현재 reload는 새 인스턴스이며 새 프로세스 검증으로 표시하지 않는다.

## GitHub 이슈와 다른 채팅에서 이어가기

`issues.json`은 모델별 한국어 이슈 초안이다. 각 초안은 고정 marker, failure signature, 버전·입력·revision, 실패 그룹의 실제 소스 위치, 관측 이유, 미확인 원인, 해결 제안, 다음 조사와 재현 명령을 포함한다. 모델 가중치·토큰·전체 환경변수는 업로드하지 않는다. private TCLMachine 저장소에 기록하는 것을 기본 운영 방식으로 권장한다. 서버의 전체 로그/그래프는 상세 증거이고, GitHub 본문 자체도 다음 채팅이 접근할 수 있는 독립 요약이다.

명시적으로 이슈를 올릴 때만 실행한다. `gh` CLI 인증이 필요하며 API/인증 실패 시 새 이슈 생성을 중단한다.

```bash
$PY "$TOOL" publish --out /root/.cache/hf-librarian/static-failures/my-new-attempt \
  --repo smallstep-zzz/TCLMachine
```

기존 marker가 있으면 이슈 댓글로 새 시도를 추가한다. 따라서 같은 모델의 개선 전/후 이력을 유지한다. GitHub 검색 결과가 중복이면 임의 선택하지 않는다. 영수증은 `github-receipts.json`에 남긴다. 같은 output을 중복 publish하지 않는다. 다음 채팅에는 이슈 URL을 주고 `result.json → traceback → preflight → graph` 순서로 확인하도록 요청한다. 공통 op/reason은 모델 여러 개에서 나타날 수 있으므로 최초 수정은 공유 규칙에서 하고, 해당 모델군을 함께 재검증한다.
