# HF Librarian for Furiosa

Hugging Face 모델 후보를 분류하고 표준 Hub 캐시에 준비하는 경량 라이브러리입니다. 모델 선정은 컴파일러 테스트의 다양성을 위한 것이며 NPU/EDF 호환성 보증이 아닙니다.

## 설치

```bash
cd /root/hf_librarian_for_furiosa
python3 -m pip install -e .
hf-librarian categories
hf-librarian status
```

Python 3.10+, `hf` 명령과 Hub 접근이 필요합니다. 인증/라이선스 승인이 필요한 모델은 미리 `hf auth login` 및 해당 모델의 사용 조건 승인이 필요합니다. 캐시 경로는 `HF_HUB_CACHE`, 기본값은 Hugging Face 기본 경로입니다. 현재 서버는 `/root/.cache/huggingface/hub`입니다. 다른 프로세스도 같은 사용자/동일 환경 설정과 모델 revision을 써야 캐시를 공유합니다.

## 사용

```bash
hf-librarian list --category language-generation
hf-librarian get --category detection
hf-librarian get --id Qwen/Qwen3-0.6B
hf-librarian prefetch --category language-generation --category segmentation
hf-librarian status
hf-librarian blacklist
```

stdout은 JSON 결과, stderr는 `missing`, `downloading`, `completed`, `available`, `failed` 이벤트입니다. 실패 시 종료 코드는 0이 아닙니다. `get`은 완료될 때까지 기다리고 snapshot 경로를 반환합니다. 이미 이 라이브러리로 준비된 파일이 존재하면 네트워크 없이 반환합니다. category는 우선순위가 가장 높은 비차단 모델 1개를 고릅니다. 명시된 모델은 목록에 먼저 추가해야 합니다.

```python
from hf_librarian import Catalog, Librarian

def event(e):
    print(e)  # callback may run on a worker thread

with Librarian(on_event=event) as lib:
    future = lib.request_async(category="language-generation")
    path = future.result()  # 예외는 여기서 발생
    path2 = lib.request(model_id="Qwen/Qwen3-0.6B")
    print(path, path2)
```

카탈로그를 다룰 때 JSON 전체를 LLM에 제공할 필요가 없습니다. `Catalog().categories()`, `.models(category, quantization="bf16")`, `.get(id)`, `.settings()`, `.version()`, `.blacklist()`, `.add_category(name, description)`, `.add_model(dict)`, `.remove_model(id)`, `.set_blacklist(id, blocked=True, reason="...")`, `.update_settings(max_concurrent_downloads=2)`를 사용하세요. CLI의 `add-category`, `add-model @model.json`, `remove-model`, `block`, `unblock`, `set`도 지원합니다. JSON5를 읽고 원자적으로 표준 JSON을 씁니다. 동시 카탈로그 편집은 지원하지 않으므로 한 작성자만 사용하세요. 설치 위치를 수정하지 않으려면 카탈로그를 복사하고 `HF_LIBRARIAN_CATALOG` 또는 `--catalog`로 지정하세요.

기본 설정은 동시 모델 다운로드 2개, 총 Hub 캐시 한도 200 GiB, 파일시스템 여유 100 GiB, 선택된 모델 파일 최대 10 GiB입니다. `Librarian(max_concurrent=4)` 또는 `prefetch --workers 4`로 호출별 조정할 수 있습니다. 이는 모델 단위 개수이며 `hf` 내부 파일 다운로드 스레드 수와 다릅니다. 다운로드 전에 Hub 메타데이터와 dry-run으로 크기를 확인하고, 여유 공간이 부족하면 중단합니다. 다른 프로그램이나 별도 Librarian 프로세스가 동시에 쓰는 양은 예약할 수 없으므로 해당 경우 한도는 엄격한 쿼터가 아닙니다. 자동 삭제는 하지 않습니다. Hub 캐시는 여러 revision을 유지할 수 있습니다.

`revision: main`은 최초 준비 시 커밋을 고정하여 이후 오프라인 재사용합니다. 최신 버전이 필요하면 모델의 revision을 새 커밋으로 변경하세요. 선택한 파일만 다운로드하므로 모델에 따라 다른 파일이나 `trust_remote_code`가 필요할 수 있습니다. 선택 파일 목록, 런타임 로더, 라이선스, 크기, 양자화 표기는 사용 전 확인하세요. 모델의 10 GiB 파일 제한은 NPU 메모리 사용량을 뜻하지 않습니다. `npu_status: untested`인 모델은 EDF 생성이나 실행을 아직 검증하지 않은 후보입니다.

## 모델 목록 갱신 스킬

`skills/hf-librarian/SKILL.md`를 지원하는 에이전트의 스킬 경로에 설치하거나 해당 파일을 읽도록 지시하세요. 신모델 조사, 외부에서 예고한 테스트의 사전 준비, 모델/카테고리 관리 절차가 있습니다. 스킬 자체는 자동 감시나 예약 작업을 만들지 않습니다.

## FXB와 FXBEX 생성 검증 및 이력

```bash
# 기본값은 PyPI 최신 버전을 확인하고, 선택한 실행 파일이 일치할 때만 실행합니다.
hf-librarian verify --id Qwen/Qwen3-0.6B --backend fxb --timeout 1800
# 다른 버전은 해당 버전이 설치된 실행 파일을 지정하고 출력 버전을 대조합니다.
hf-librarian verify --id Qwen/Qwen3-0.6B --backend fxb --executable /usr/local/bin/fxb --version 2026.4.0
hf-librarian history --id Qwen/Qwen3-0.6B
```

기본 실행 파일은 PATH의 `fxb`/`fxbex`이며 기본 동작은 PyPI에서 최신 `furiosa-llm`/`fxbex` 버전을 조회하고 일치하지 않으면 중단합니다. SDK를 자동 설치·업그레이드하지 않습니다. 사내 SDK 인덱스가 PyPI와 다른 경우 최신 여부를 따로 확인하고 `--executable`과 `--version`을 명시하세요. 해당 버전의 실제 실행 파일이 필요하며, 버전 문자열만 바꿔 다른 버전을 실행할 수 없습니다. FXB 기록의 `actual_version`은 `fxb` 버전이며, 확인 가능한 경우 `compiler_version`에 `furiosa-tcc` 버전이 따로 기록됩니다. 조회할 수 없으면 null입니다.

공식 `fxb build`는 캐시된 모델의 snapshot을 사용하고, 성공 판정에는 실제 파일과 `fxb show` 성공이 필요합니다. 컴파일 옵션은 `--build-arg=--max-model-len --build-arg=512`처럼 전달합니다. 무거운 작업일 수 있으므로 시간과 디스크 여유를 고려하세요.

FXBEX는 Hugging Face 모델을 직접 컴파일하지 않고, 별도로 만들어진 컴파일 디렉터리를 패키징합니다. 준비된 디렉터리가 있고 `manifest.json`의 `provenance.source_url` 및 `provenance.revision`이 이 모델과 캐시 revision에 일치할 때 사용합니다.

```bash
hf-librarian verify --id Qwen/Qwen3-0.6B --backend fxbex --source /path/to/compiled-directory
```

FXBEX 결과는 `fxbex inspect`를 통과해야 성공입니다. 준비 입력이 없거나 버전이 다르면 `blocked`, 빌드·검사 실패는 `failed`, 제한 시간 초과는 `timeout`으로 이력에 남습니다. 결과와 로그, 생성 파일, SHA-256, 입력 revision, 버전 및 명령은 기본 `~/.cache/hf-librarian/verification`에 저장합니다. `--results-dir` 또는 `HF_LIBRARIAN_RESULTS`로 변경할 수 있습니다. 모델 실행·정확성·NPU 트레이스는 이 기능이 검증하지 않습니다. 버전별 이력은 삭제하지 않고 누적하며 캐시나 GitHub에는 무거운 결과 파일을 넣지 않습니다.

```python
from hf_librarian import Librarian, Verifier
with Librarian() as lib:
    verifier = Verifier(lib)
    result = verifier.verify('Qwen/Qwen3-0.6B', backend='fxb', version='2026.4.0')
    print(result['status'], result['actual_version'], result.get('compiler_version'))
    print(verifier.history('Qwen/Qwen3-0.6B'))
```
