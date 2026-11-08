# 방법 비교 벤치마크 — 다른 컴퓨터에서 남은 인스턴스 계산하기

지시서: `PROMPT_method_comparison.md`. 메인 PC(Taeuk_Desktop)는 n4_s0, n4_s1, n6_s0, n7_s*, n8_s* 를 계산한다.
**이 문서의 컴퓨터는 아래 6개 인스턴스 전체(모든 방법)를 계산하고 결과 JSON 을 git 으로 돌려준다.**
한 인스턴스의 모든 방법은 반드시 같은 컴퓨터에서 돌린다(시간 비교의 공정성).

| 담당 | 인스턴스 (파편) |
|---|---|
| 이 컴퓨터 | n6_s2, n6_s1, n5_s2, n5_s1, n5_s0, n4_s2 (`bench/instances.json`) |

> **변경 (2026-11-09): DP(M4)와 M3 시간 예산 설정(M3time)은 비교에서 뺐다.**
> 지금 M4 나 M3time 을 계산 중인 프로세스(도우미 포함)는 멈춘다. `git pull` 후, 아직 M0·M5·M1·M2·M3gen 이 남은 인스턴스만 아래 명령으로 다시 띄운다.
> 다시 띄워도 M0·M5 는 체크포인트에서 이어 풀고, 끝난 방법은 건너뛴다.
> 이미 생긴 `*__M4.json`, `*__M3_time_s*.json` 은 지우지 않아도 된다(보고서에서 제외).

## 1. 준비 (한 번)

1. **Python 3.10+**
   ```
   git clone https://github.com/gramschmidtz/TD_route_planning.git
   cd TD_route_planning
   git checkout spacemission
   python -m pip install -r bench/requirements.txt
   ```
2. **IBM ILOG CPLEX 22.x 설치 필수.** CBC 로 대체하지 않는다. 경로가 메인 PC 와 다르면 환경변수로 준다.
   - PowerShell: `$env:CPLEX_PATH = "C:\Program Files\IBM\ILOG\CPLEX_Studio222\cplex\bin\x64_win64\cplex.exe"`
   - bash: `export CPLEX_PATH=/opt/ibm/ILOG/CPLEX_Studio222/cplex/bin/x86-64_linux/cplex`
3. **데이터:** `external/adr_time_varying_benefit/adr/out_0605/` 와 `data_0605/` 가 있어야 한다.
   - 저장소에 없으면 메인 PC 의 같은 폴더를 그대로 복사한다(`adr_time_varying_benefit.zip` 압축 해제본).
   - 설정은 `config/adr_fy1c_20260605.yaml` 이다.
4. **설치 확인 (1–2분):**
   ```
   python -m pytest -q tests/test_adr_data.py
   ```

## 2. 기계 보정 (먼저, 약 5–10분)

메인 PC 와 시간을 비교할 수 있도록 n4_s0 의 M0 를 이 컴퓨터에서 한 번 더 푼다.
결과는 `bench/out/fronts/n4_s0__CALIB_<호스트>_M0.json` 으로 저장된다.
```
python -W ignore -m bench.run --calib
```

## 3. 본 계산

규칙:
- 인스턴스 하나당 프로세스 하나.
- CPLEX 는 1스레드(기본값). 동시에 띄우는 프로세스 수는 **물리 코어 수 − 2 이하**로 한다(6개면 물리 코어 8개 이상).
- 계산 중에는 다른 무거운 작업을 하지 않는다.

**인스턴스 하나당 프로세스 하나(6개).** 도우미는 더 이상 필요 없다(DP·M3 시간 예산 설정을 뺐기 때문).

PowerShell (같은 창에서 `$env:CPLEX_PATH` 설정 후, **한 번만** 실행):
```
foreach ($i in "n6_s2","n6_s1","n5_s2","n5_s1","n5_s0","n4_s2") {
  Start-Process -NoNewWindow -FilePath python -ArgumentList "-W ignore -m bench.run --inst $i --methods M0 M5 M1 M2 M3gen" -RedirectStandardOutput "bench_remote_$i.log" -RedirectStandardError "bench_remote_$i.err"
}
```
실행 중인 bench 프로세스 수 확인 (6 이어야 한다):
```
(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*bench.run*' }).Count
```

방법 순서와 대략의 양(인스턴스 하나당):

| 방법 | 내용 |
|---|---|
| M0 | 참 프론트, 점수 축 exact ε. 수십 분–수 시간 |
| M5 | 대기 없는 모델 |
| M3gen | NSGA-II pop 200 × 250세대, 시드 10개. 각 수 초 |
| M1 | 가중합, λ 11개와 101개 |
| M2 | ΔV 격자, δ = 100 / 50 / 10 |

**중단·재시작:** 같은 명령을 다시 실행하면 이어서 계산한다.
- 끝난 방법은 `bench/out/fronts/<인스턴스>__<방법>.json` 이 있으면 건너뛴다.
- M0·M5 는 `%LOCALAPPDATA%\td_route_bench_work\<인스턴스>\` 의 체크포인트에서 이어 푼다.

진행 확인: `bench_remote_*.log` (방법 하나가 끝날 때마다 한 줄).

## 4. 결과 돌려주기

결과는 `bench/out/fronts/*.json` 뿐이다(인스턴스·방법별 프론트 + 시간·계산량·메모리 + `machine` 정보).
중간에 끝난 것만 보내도 된다. 메인 PC 는 이미 있는 파일을 다시 계산하지 않는다.
```
git checkout -b bench-remote-<호스트이름>
git add bench/out/fronts/
git commit -m "bench: remote results (<호스트이름>)"
git push -u origin bench-remote-<호스트이름>
```
메인 PC 에서는 이 브랜치의 `bench/out/fronts/` 만 가져와 `python -W ignore -m bench.report` 로 지표·그림·요약표를 만든다.
