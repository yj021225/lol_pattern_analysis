<p align="center">
<img src="./img/lol_logo.png"/>
</p>

# lol pattern analysis
<!-- 
badge icon 참고 사이트
https://github.com/danmadeira/simple-icon-badges
-->
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />

## 1. 개요
### 1.1 리그 오브 레전드 게임 분석의 영향력
LoL은 글로벌 e스포츠 시장의 중심에 있는 게임으로, 분석을 통해 팀의 전략 개선에 기여할 수 있습니다.<br>
LoL에서 생성되는 방대한 데이터는 데이터 분석, 머신 러닝 및 AI 알고리즘 테스트에 중요한 자원이 됩니다.<br>
예측 모델 구축, 추천 시스템 개발, 게임 밸런스 최적화 등에 활용할 수 있습니다.<br>
### 1.2 문제 정의
나는 이번 프로젝트를 통해 미드라이너 데이터를 전처리하여 미드라이너의 실력지표 산출 및 평가를하여<br>
승리할 확률을 예측하는 모델을 구축할 것이다. 이번 프로젝트를 통해 E-sports산업에 도움이 되었으면 한다.<br>

## 2. 데이터 전처리
### 2.1 성능지표 구축
[extracted_full_data.json](./extracted_full_data.json)<br>
[extracted_full_data_o.json](./extracted_full_data_o.json)<br>
[extracted_test_data.json](./extracted_test_data.json)<br>
[extracted_test_opponent_data.json](./extracted_test_opponent_data.json)<br>
미드라이너 데이터를 통해 14분이전과 14분이후로 나누어 성능 지표를 산출하는 성능지표를 구축했다.<br>
```
[
    {
        "gamerName": "龙宫公主",
        "numValidGame": 58,
        "match": [
            {
                "riotIdGameName": "龙宫公主",
                "matchId": "KR_7250671169",
                "gameCreation": 1725030458253,
                .
                .
                .
                "at14": {
                    "kills": 2,
                    "deaths": 1,
                    "assists": 1,
                    .
                    .
                    .
                },
                "af14": {
                    "kills": 8,
                    "deaths": 4,
                    "assists": 14,
                    .
                    .
                    .
                }
            },
```
extracted_full_data.json
### 2.2 비교 성능지표 구축
[merged_data_full.json](./merged_data_full.json)<br>
[merge_data_sample.json](./merge_data_sample.json)<br>
플레이어와 상대 간의 지표 비교를 통해 초반 및 후반 데이터를 통합적으로 분석할 수 있는 구조를 구축했다.<br>
```
[
    {
        "GamaName": "龙宫公主",
        "matches": [
            {
                "matchId": "KR_7250671169",
                "gameCreation": 1725030458253,
                .
                .
                .
                "at14": {
                    "gameDuration": 840.239,
                    "combat": {
                        "killsRatio": 0.4,
                        .
                        .
                        .
                },
                "af14": {
                    "gameDuration": 1021.033415313,
                    "combat": {
                        "killsRatio": 0.8,
                        .
                        .
                        .
            },
```
merged_data_full.json
### 2.3 학습 데이터 구축
[final_target_data.json](./final_target_data.json)<br>
병합된 데이터에서 핵심 지표를 추출하여 데이터 분석 및 모델 학습에 적합한 형식으로 가공하여 학습 데이터를 구축했다.<br>
```
[
    {
        "gamerName": "龙宫公主",
        "opponentGamerName": "n8jj",
        "matchID": "KR_7250671169",
        "gameCreation": 1725030458253,
        .
        .
        .
    },
```
final_target_data.json
## 3.미드라이너 실력 지표 산출 및 평가
```
naive_analysis_at14
모델 계수 (theta) :  [ 2.24044148e-01 -1.59028323e-01  1.73470044e-01 -4.73071095e-02
  5.23509301e-02  1.43965122e-04 -5.02227294e-05]
모델 절편 (b) :  0.37967548207258306
정확도 (Accuracy, threshold: 0.5): 0.6141160949868074
AUROC: 0.644451501358456
```
이 결과는 14분 이전의 결과로 초반 결과라고 볼 수 있다.<br>
모델 계수의 결과를 보면 초반 킬 비율과 어시스트 비율이 승리에 가장 큰 영향을 주는 반면, 데스 비율은 부정적 영향을 미칩니다.<br>
정확도를 보았을 때 0.61로 0.5보다는 높아 유의미한 결과라고 볼 수 있지만 그리 높은 결과는 아니라고 볼 수 있다.<br>
최종적으로 승부를 예측하는 AUROC 결과를 보면 0.64로 무작위 추측보다는 높지만 완전히 예측한다고 보기는 어렵다.<br>
```
naive_analysis_af14
모델 계수 (theta) :  [ 3.61414532e-01 -4.59612863e-01  7.37365400e-01  9.15368538e-02
 -7.88146861e-03  2.78917766e-05 -6.38308498e-05]
모델 절편 (b) :  0.18431338828950838
정확도 (Accuracy, threshold: 0.5): 0.8700527704485488
AUROC: 0.9416675487809181
```
이 결과는 14분 이후의 결과이다. 14분 이전의 결과와 많이 달라진 것을 알 수 있다.<br>
모델 계수의 결과를 보면 어시스트 비율, 데스 비율, 킬 비율이 승리 여부에 가장 큰 영향을 미칩니다.<br>
초반 데미지 관련 지표(DPM, DTPM)는 영향력이 낮습니다.<br>
정확도는 0.87로 매우 높은 정확도를 보여주며 초반 데이터와 달리 예측을 잘하는 것으롷 볼 수 있다.<br>
AUROC 점수가 0.9 이상이면 매우 좋은 성능을 나타냅니다.<br>
이 점수는 모델이 승리와 패배를 거의 완벽하게 구분할 수 있음을 보여줍니다.<br>
