# PRD улучшения — rag-eval-harness

> Харнесс должен доказывать свой тезис на самом себе: метрики обязаны **мочь упасть** и **различать слои**, а не воспроизводить число, зашитое в конструкцию.

Owner: Никита · Составлен: 2026-10-14 · Область: `rag-eval-harness/` · Видимость: public · Кейс сайта: `projects/rag.md`

## Контекст

Член реестра `rag-eval-harness` (`kind: member`, `visibility: public`, `on_site: true`, case `rag`). Офлайн-харнесс оценки RAG: TF-IDF-ретрив по 12 концепт-докам, детерминированный экстрактивный ответчик, метрики ретрива (recall@k, MRR, precision@k) и генерации (faithfulness, correctness) на 20 размеченных вопросах. Продающий тезис кейса: «измерь, а не демонстрируй; видно, какой слой улучшать». Источник истины процедуры — `<portfolio>/LOOP.md`; инвариант — правда → глубина → нарратив.

## Текущее состояние (доказательства, не мнения)

| Ось | Что сейчас | Доказательство |
|---|---|---|
| Правда | faithfulness=1.00 подан как результат, но **тавтологичен**: экстрактивный ответ — verbatim-подстрока top-дока, faithfulness = доля токенов ответа в контексте | `src/generate.py:34-49`; `src/eval.py:44-55`; прогон: faithfulness=1.0 на всех 20, ответ-подстрока 20/20; тест `tests/test_rag.py:29,38` |
| Правда | Ни одно число rag не закреплено claim-ом | `grep -i rag scripts/check_claims.py` → пусто; `make check` = 49 claims, rag не покрыт |
| Правда | README заявляет `reports/` gitignored — файлы **закоммичены** | `README.md:111` vs `git ls-files reports` → 3 файла; `.gitignore` их не содержит |
| Глубина | Центральный тезис не подтверждается: обе headline-метрики вырождены | recall@5 тривиален (12 доков, top-5 = 42% корпуса; `src/eval.py:35`), faithfulness тавтологичен, precision@5=0.2 детерминирован |
| Глубина | `recall@k` реализован как hit-rate (`any(hits)`), не доля найденных gold | `src/eval.py:35`; сейчас gold=1 для всех 20 (`data/eval.jsonl`) |
| Глубина | `SEED=42` — мёртвый конфиг | `src/config.py:13` (никаких ссылок) |
| Нарратив | README и docstring обещают CI, workflow отсутствует | `README.md:31`; `src/generate.py:6`; `.github/` нет |
| Упаковка | `uv.lock` в `.gitignore` → не закоммичен | `.gitignore:5`; `git ls-files uv.lock` → пусто |
| Упаковка | `anthropic>=0.40` — обязательная зависимость при опциональном LLM | `pyproject.toml:14` vs `src/generate.py:53` (ленивый импорт); Quickstart ставит без anthropic `README.md:90-94` |
| Упаковка | Пакет назван `src` (top-level) | `pyproject.toml:21`; импорты `from src.index` пронизывают код |

Preflight: `bash scripts/preflight.sh` → `preflight: OK` (49 claims, registry OK). Тесты члена: `5 passed`.
Числа воспроизведены: `recall@k 1.0, mrr 1.0, precision@k 0.2, faithfulness 1.0, correctness 0.369`.

## Цели и метрики (baseline → target)

| Метрика | Baseline | Target | Команда воспроизведения |
|---|---|---|---|
| headline-метрик, которые «не могут упасть» | 1 (faithfulness) | 0 | `uv run --no-project --quiet --with scikit-learn --with pandas --with numpy python -c "from src.eval import run_eval; print(run_eval(True,5)['summary'])"` |
| headline-чисел rag под claim-ом | 0 из 5 | ≥4 из 5 | `bash scripts/preflight.sh` (число claims растёт, rag присутствует) |
| негативный контроль различает ретрив | нет | recall@k control < 1.0 на перестановке/шумовом ретривере | новый тест `tests/` (REQ-002) |
| `.github/workflows` у члена | 0 | ≥1 (тесты зелёные на PR) | `gh run list --repo NikitaBoyarkin/rag-eval-harness` |

## Требования

### REQ-001 — Снять тавтологию faithfulness из нарратива и теста (P1, High, S)

**Находка:** Правда · faithfulness=1.00 — метрика, которая не может упасть · **Доказательство:** `src/generate.py:34-49` + `src/eval.py:44-55` (ответ ⊂ контекст ⇒ faithfulness≡1.0; прогон 20/20), `tests/test_rag.py:29,38`, `README.md:3,37`, `NikitaBoyarkin.github.io/src/content/projects/rag.md:3,8,14,55`.
**Проблема:** число подано как результат слоя генерации, хотя это свойство конструкции (extractive ⇒ grounded by definition). Рекрутерский read: метрика-фикция.
**Требуется:** в README и кейсе пометить offline-faithfulness как **прокси, вырожденный by construction** (1.0 не является измерением), убрать `1.00` из блока headline-результатов; тест переформулировать так, чтобы он фиксировал **инвариант конструкции**, а не «качество» (комментарий + имя).
**Acceptance criteria:**
1. `README.md:3` и кейс сайта `NikitaBoyarkin.github.io/src/content/projects/rag.md:3,8,14,55` не содержат faithfulness=1.00 в блоке headline-результатов.
2. `tests/test_rag.py:29,38` — тест назван/комментирован как «invariant: extractive answer is grounded by construction» (не как качество).
3. В `Notes`/`Ограничения` явно сказано: offline-faithfulness вырожден, измеряется только с `--llm`.
**Проверка:** `rg -n 'faithfulness' README.md` → нет строки-результата со значением 1.00; `pytest -q` зелёный.
**Исполнитель:** @report-styling (проза) + @test-driven-development (переименование теста).

### REQ-002 — Метрики обязаны различать слои: негативный контроль (P1, High, M) — требует решения владельца

**Находка:** Глубина · центральный тезис харнесса не подтверждается · **Доказательство:** `src/eval.py:35`, `README.md:41-44,82`, `NikitaBoyarkin.github.io/src/content/projects/rag.md:74`; recall@5 тривиален при 12 доках, faithfulness тавтологичен, precision@5 детерминирован.
**Проблема:** харнесс продаёт «видно, какой слой улучшать», но на текущем корпусе ни одна метрика не может показать failure mode. Это подрывает единственную ценность проекта.
**Требуется:** добавить в тесты **негативный контроль**, доказывающий дискриминативность метрик, — малый, детерминированный, офлайн. Предпочтительный вариант (выбрать один, отметить в PRD-ревью):
- (A) baseline-нулевая модель: перемешанный/рандом-rank ретривер должен давать recall@k строго < 1.0 (и падать ниже 1.0 при k меньше числа доков);
- (B) расширить корпус/набор так, чтобы у части вопросов gold не попадал в top-1 (тогда MRR < 1.0 — метрика что-то измеряет).
**Acceptance criteria:**
1. Новый тест доказывает: при деградации ретривера `recall@k`/`mrr` **падают** (assert строго `<` порога).
2. Тест детерминированный и офлайн (без ключей), входит в `pytest -q`.
3. README/кейс объясняют разницу «на этом корпусе ретрив идеален» vs «метрика различает деградацию» (это и есть доказательство ценности).
**Проверка:** `uv run --no-project --quiet --with scikit-learn --with pandas --with numpy --with pytest pytest -q` → новый тест зелёный и падает при подмене ретривера.
**Исполнитель:** @ab-test-methodology-scripts (негативный контроль/калибровка) + @test-driven-development.

### REQ-003 — Закрепить числа rag claim-ом в `scripts/check_claims.py` (P1, Med, S)

**Находка:** Правда · дрейф чисел ничем не ловится · **Доказательство:** `grep -i rag scripts/check_claims.py` → пусто; `make check` = 49 claims без rag.
**Проблема:** headline-числа rag (recall@k 1.0, mrr 1.0, precision@k 0.2, correctness 0.369) не привязаны к данным — гейт правды их не защищает.
**Требуется:** добавить `rag_claims()` по образцу соседних функций (`scripts/check_claims.py:351` `CLAIMS`), пересчитывающую числа из `data/eval.jsonl` + корпуса.
**Acceptance criteria:**
1. `bash scripts/preflight.sh` включает rag-строки и печатает их как `OK`.
2. Значение `expected` берётся из README с цитатой `file:line`, **не** из только что посчитанного (анти-тавтология — грабли из `@portfolio-cycle-run`).
3. gейт падает при подмене README-числа (мутационная проверка).
**Проверка:** `/Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python scripts/check_claims.py` → rag присутствует, OK.
**Замечание по реализации:** `check_claims` идёт на EDA-venv без scikit-learn (Makefile `eda-venv`); пересчёт метрик требует `uv run --with scikit-learn …` подпроцессом **или** добавления `scikit-learn` в `eda-venv`. Решить при исполнении.
**Исполнитель:** @context-architecture.

### REQ-004 — CI-workflow для члена (P2, Med, S)

**Находка:** Нарратив/Упаковка · обещание CI не подкреплено · **Доказательство:** `README.md:31`, `src/generate.py:6`, `.github/` отсутствует (`ci: нет` в recon); тесты есть и идут 59с. Контекст: CI настроен у 17 членов; rag — один из 8 членов с `tests/` без CI (`causal-uplift`, `clickhouse-analytics`, `gmv-drop-rca`, `ml`, `python`, `social-network-analytics`, `streamlit-app`).
**Требуется:** добавить `.github/workflows/ci.yml`, прогоняющий `pytest -q` (и, если правка REQ-003 попадёт в CI, — claims), офлайн, без ключей.
**Acceptance criteria:**
1. Workflow зелёный на push/PR.
2. В нём нет секретов; шаг не требует `ANTHROPIC_API_KEY`.
3. README-строка про CI становится правдой (или снимается).
**Проверка:** `gh run list --repo NikitaBoyarkin/rag-eval-harness --limit 1` → success.
**Исполнитель:** @ci-cd-and-automation.

### REQ-005 — Воспроизводимость зависимостей: закоммитить `uv.lock` (P2, Med, S)

**Находка:** Упаковка · нет залоченных зависимостей на чистом клоне · **Доказательство:** `.gitignore:5` (`uv.lock`), `git ls-files uv.lock` пусто.
**Требуется:** убрать `uv.lock` из `.gitignore` и закоммитить; либо осознанно оставить и убрать из питча слово «reproducible» без оговорки.
**Acceptance criteria:**
1. `git ls-files uv.lock` непусто.
2. `uv sync` на чистом клоне даёт те же версии (lock присутствует).
**Проверка:** `git ls-files uv.lock`; `uv lock --check`.
**Исполнитель:** @python-patterns.

### REQ-006 — `anthropic` — опциональная зависимость (P2, Med, S)

**Находка:** Упаковка · объявленная зависимость противоречит фактическому запуску · **Доказательство:** `pyproject.toml:14` (обязательна) vs `src/generate.py:53` (ленивый импорт) vs `README.md:90-94` (Quickstart без anthropic).
**Требуется:** перенести `anthropic` в `[project.optional-dependencies]` (напр. `llm = [...]`), согласовать Quickstart и pyproject.
**Acceptance criteria:**
1. Базовый `uv sync` не тянет anthropic; ядро и тесты зелёные без него.
2. Запуск с `--llm` документирован через extra.
3. Quickstart и pyproject не расходятся.
**Проверка:** `uv run --no-project --quiet python -c "import tomllib,pathlib;d=tomllib.loads(pathlib.Path('pyproject.toml').read_text());print('anthropic' in d['project']['dependencies'])"` → `False`.
**Исполнитель:** @python-patterns.

### REQ-007 — Честная семантика `recall@k` (P2, Med, S)

**Находка:** Глубина · метрика названа recall, считает hit-rate · **Доказательство:** `src/eval.py:35` (`any(hits)`); все 20 вопросов имеют ровно 1 gold (`data/eval.jsonl`).
**Требуется:** либо переименовать в `hit_rate@k` (честно отражает семантику), либо реализовать настоящий `recall@k = |retrieved∩gold|/|gold|` (устойчив к multi-gold).
**Acceptance criteria:**
1. Имя метрики совпадает с семантикой (в коде, README, кейсе, `reports/`).
2. Добавлен юнит-тест на multi-gold вопрос (recall < 1 при неполном извлечении).
**Проверка:** `pytest -q`; `rg -n 'recall@k' src/ tests/` согласовано.
**Исполнитель:** @ab-test-analysis.

### REQ-008 — Убрать ложный комментарий про gitignored `reports/` (P3, Low, S)

**Находка:** Правда · комментарий противоречит ФС · **Доказательство:** `README.md:111` vs `git ls-files reports` → 3 файла.
**Требуется:** заменить «(gitignored)» на «(committed)» либо реально заигнорить и снять указатели.
**Acceptance criteria:**
1. `README.md:111` соответствует `git ls-files` состоянию.
**Проверка:** `rg -n 'reports/' README.md` без слова gitignored.
**Исполнитель:** @report-styling.

### REQ-009 — Убрать мёртвый `SEED` (P3, Low, S)

**Находка:** Глубина · конфиг не используется · **Доказательство:** `src/config.py:13` (нет ссылок в `src/`, `run.py`, `tests/`).
**Требуется:** удалить `SEED` (детерминизм обеспечивают TF-IDF + extractive) или начать использовать.
**Acceptance criteria:**
1. `rg -n 'SEED' src/ run.py tests/` → пусто.
2. Тесты зелёные.
**Проверка:** `pytest -q`.
**Исполнитель:** @python-patterns.

### REQ-010 — Семантика имени пакета `src` (P4, Low, M) — опционально

**Находка:** Упаковка · top-level пакет `src` = риск коллизии при установке · **Доказательство:** `pyproject.toml:21`, импорты `from src.*`.
**Требуется:** (опционально) переименовать пакет в `rag_eval` и обновить импорты/`hatch`-таргет; либо задокументировать, что пакет не публикуется в PyPI.
**Acceptance criteria:**
1. Либо `import rag_eval` работает и `uv build` собирает корректный wheel; либо в README прямо сказано «не публикуется».
**Проверка:** `uv build` + `uv run python -c "import rag_eval"`.
**Исполнитель:** @python-patterns.

## Вне объёма

- Перенос харнесса на эмбеддинги/реальный LLM — осознанный swap-point (`README.md:128-130`), не дефект этого PRD.
- Замена `correctness` (Jaccard) на LLM-as-judge — задокументированное ограничение, не улучшение здесь.
- Правка общего корневого `scripts/check_claims.py` сверх добавления rag-функции — вне объёма.

## HUMAN-гейты (нужен человек)

| # | Что | Почему нельзя автономно |
|---|---|---|
| REQ-002 | Выбор подхода к дискриминативности метрик (A негативный контроль vs B расширение корпуса) | Меняет метод/объём данных члена — дизайн-решение владельца, не механика |
| REQ-010 | Переименование пакета `src` → `rag_eval` | Трогает импорты по всему проекту; решение об объёме — за владельцем |

Правка `scripts/check_claims.py` (REQ-003), `.gitignore` (REQ-005), снятие ложного комментария (REQ-008) — **автономная зона** (`LOOP.md`: claims/прозы, gate остаётся зелёным). Удалений, приватных членов, `git push` и правки доктрины (`docs/adr/`, `LOOP.md`) в этом PRD нет.

## Риски

- **REQ-002 меняет метрики** → может сдвинуть headline-числа → синхронно обновить README + кейс + (новые) claims (REQ-003) в одном цикле, иначе дрейф.
- **REQ-003 требует scikit-learn в EDA-venv** → не расширять venv молча; решить (подпроцесс `uv run` vs venv-пакет) до исполнения.
- **REQ-005 закоммитит lock** → проверить, что он не тянет лишних платформенных зависимостей.

## Handoff

- Исполнение: `@portfolio-cycle-run` (правки по бэклогу, cap N=3, локальные коммиты; `make check` зелёный до и после) или `@prd-taskmaster` (разбить REQ на задачи).
- Верхушка следующего цикла: **REQ-001 → REQ-003 → REQ-002** (правда → правда → глубина), каждый с тестом; REQ-004/REQ-005 — сразу за ними как «упаковка».
- Запись в общий бэклог: строки пересекаются с `reports/loop/BACKLOG.md` B36 (faithfulness не может упасть) и B7 (упаковка rag) — обновить их статус, не дублировать.
- Перепроверка: `bash scripts/preflight.sh` + `pytest -q` в члене собственной командой.
