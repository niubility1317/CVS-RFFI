# LocalRidge联合通道adapter support pilot

完整160 parent、4 row、三路径均已核验；旧类固定6个。A与B−A为N/A。R_channel_seq为预声明顺序主线，R_channel_reset仅作继承对照。

| 诊断 | 路径 | K | 新类数 | A旧 | B0旧 | B旧 | C旧类列 | C旧 | C新 | H | B−B0 | B−C旧类列 | 新竞争损失 | 注册下降 | 新旧差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | R0 | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 5 | 0 | N/A | 63.333 | 63.333 | 63.333 | 63.333 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R0 | 5 | 2 | N/A | 63.333 | 63.333 | 63.333 | 61.250 | 50.000 | 53.522 | 0.000 | 0.000 | 2.083 | 2.083 | 17.083 |
| oof | R0 | 5 | 5 | N/A | 63.333 | 63.333 | 62.500 | 54.583 | 50.000 | 51.918 | 0.000 | 0.833 | 7.917 | 8.750 | 7.083 |
| oof | R0 | 5 | 10 | N/A | 63.333 | 63.333 | 63.750 | 53.333 | 48.250 | 50.361 | 0.000 | -0.417 | 10.417 | 10.000 | 8.083 |
| oof | R0 | 5 | 20 | N/A | 63.333 | 63.333 | 63.750 | 48.750 | 46.250 | 46.539 | 0.000 | -0.417 | 15.000 | 14.583 | 10.500 |
| oof | R0 | 10 | 0 | N/A | 73.542 | 73.542 | 73.542 | 73.542 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R0 | 10 | 2 | N/A | 73.542 | 73.542 | 73.958 | 71.667 | 51.875 | 59.451 | 0.000 | -0.417 | 2.292 | 1.875 | 20.208 |
| oof | R0 | 10 | 5 | N/A | 73.542 | 73.542 | 73.125 | 67.917 | 56.500 | 61.409 | 0.000 | 0.417 | 5.208 | 5.625 | 11.833 |
| oof | R0 | 10 | 10 | N/A | 73.542 | 73.542 | 73.542 | 65.625 | 57.875 | 61.456 | 0.000 | 0.000 | 7.917 | 7.917 | 8.083 |
| oof | R0 | 10 | 20 | N/A | 73.542 | 73.542 | 73.542 | 64.167 | 56.500 | 60.024 | 0.000 | 0.000 | 9.375 | 9.375 | 7.750 |
| oof | R0 | 20 | 0 | N/A | 76.250 | 76.250 | 76.250 | 76.250 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R0 | 20 | 2 | N/A | 76.250 | 76.250 | 76.875 | 74.688 | 53.125 | 61.590 | 0.000 | -0.625 | 2.188 | 1.562 | 21.562 |
| oof | R0 | 20 | 5 | N/A | 76.250 | 76.250 | 76.875 | 70.104 | 61.250 | 65.085 | 0.000 | -0.625 | 6.771 | 6.146 | 8.854 |
| oof | R0 | 20 | 10 | N/A | 76.250 | 76.250 | 76.771 | 68.021 | 63.625 | 65.703 | 0.000 | -0.521 | 8.750 | 8.229 | 4.396 |
| oof | R0 | 20 | 20 | N/A | 76.250 | 76.250 | 77.083 | 65.104 | 62.906 | 63.944 | 0.000 | -0.833 | 11.979 | 11.146 | 3.448 |
| oof | R_channel_reset | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_reset | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_reset | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_reset | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_reset | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_reset | 5 | 0 | N/A | 63.333 | 62.917 | 62.917 | 62.917 | N/A | N/A | -0.417 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_channel_reset | 5 | 2 | N/A | 63.333 | 62.917 | 63.333 | 61.667 | 48.750 | 52.507 | -0.417 | -0.417 | 1.667 | 1.250 | 18.750 |
| oof | R_channel_reset | 5 | 5 | N/A | 63.333 | 62.917 | 62.500 | 55.417 | 49.000 | 51.636 | -0.417 | 0.417 | 7.083 | 7.500 | 8.750 |
| oof | R_channel_reset | 5 | 10 | N/A | 63.333 | 62.917 | 63.750 | 53.750 | 48.250 | 50.594 | -0.417 | -0.833 | 10.000 | 9.167 | 7.667 |
| oof | R_channel_reset | 5 | 20 | N/A | 63.333 | 62.917 | 64.167 | 48.750 | 46.375 | 46.661 | -0.417 | -1.250 | 15.417 | 14.167 | 9.875 |
| oof | R_channel_reset | 10 | 0 | N/A | 73.542 | 74.375 | 74.375 | 74.375 | N/A | N/A | 0.833 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_channel_reset | 10 | 2 | N/A | 73.542 | 74.375 | 73.750 | 71.458 | 52.500 | 59.900 | 0.833 | 0.625 | 2.292 | 2.917 | 18.958 |
| oof | R_channel_reset | 10 | 5 | N/A | 73.542 | 74.375 | 73.542 | 68.125 | 56.250 | 61.325 | 0.833 | 0.833 | 5.417 | 6.250 | 12.292 |
| oof | R_channel_reset | 10 | 10 | N/A | 73.542 | 74.375 | 73.125 | 66.667 | 58.125 | 62.067 | 0.833 | 1.250 | 6.458 | 7.708 | 8.542 |
| oof | R_channel_reset | 10 | 20 | N/A | 73.542 | 74.375 | 74.583 | 64.167 | 56.750 | 60.152 | 0.833 | -0.208 | 10.417 | 10.208 | 7.750 |
| oof | R_channel_reset | 20 | 0 | N/A | 76.250 | 76.562 | 76.562 | 76.562 | N/A | N/A | 0.313 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_channel_reset | 20 | 2 | N/A | 76.250 | 76.562 | 76.875 | 74.583 | 53.438 | 61.663 | 0.313 | -0.312 | 2.292 | 1.979 | 21.146 |
| oof | R_channel_reset | 20 | 5 | N/A | 76.250 | 76.562 | 76.562 | 70.208 | 61.125 | 65.022 | 0.313 | -0.000 | 6.354 | 6.354 | 9.083 |
| oof | R_channel_reset | 20 | 10 | N/A | 76.250 | 76.562 | 76.562 | 67.917 | 64.312 | 66.012 | 0.313 | 0.000 | 8.646 | 8.646 | 3.812 |
| oof | R_channel_reset | 20 | 20 | N/A | 76.250 | 76.562 | 77.188 | 65.833 | 63.844 | 64.772 | 0.313 | -0.625 | 11.354 | 10.729 | 3.510 |
| oof | R_channel_seq | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_seq | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_seq | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_seq | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_seq | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_channel_seq | 5 | 0 | N/A | 63.333 | 62.917 | 62.917 | 62.917 | N/A | N/A | -0.417 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_channel_seq | 5 | 2 | N/A | 63.333 | 62.917 | 62.917 | 60.833 | 48.750 | 52.011 | -0.417 | 0.000 | 2.083 | 2.083 | 19.583 |
| oof | R_channel_seq | 5 | 5 | N/A | 63.333 | 62.917 | 62.083 | 54.583 | 49.000 | 51.294 | -0.417 | 0.833 | 7.500 | 8.333 | 8.750 |
| oof | R_channel_seq | 5 | 10 | N/A | 63.333 | 62.917 | 63.750 | 54.167 | 48.500 | 50.933 | -0.417 | -0.833 | 9.583 | 8.750 | 7.833 |
| oof | R_channel_seq | 5 | 20 | N/A | 63.333 | 62.917 | 63.750 | 50.417 | 46.250 | 47.477 | -0.417 | -0.833 | 13.333 | 12.500 | 10.000 |
| oof | R_channel_seq | 10 | 0 | N/A | 73.542 | 74.375 | 74.375 | 74.375 | N/A | N/A | 0.833 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_channel_seq | 10 | 2 | N/A | 73.542 | 74.375 | 73.542 | 71.250 | 53.750 | 60.438 | 0.833 | 0.833 | 2.292 | 3.125 | 19.583 |
| oof | R_channel_seq | 10 | 5 | N/A | 73.542 | 74.375 | 73.750 | 67.917 | 56.250 | 61.244 | 0.833 | 0.625 | 5.833 | 6.458 | 12.083 |
| oof | R_channel_seq | 10 | 10 | N/A | 73.542 | 74.375 | 73.333 | 66.667 | 58.500 | 62.295 | 0.833 | 1.042 | 6.667 | 7.708 | 8.167 |
| oof | R_channel_seq | 10 | 20 | N/A | 73.542 | 74.375 | 74.167 | 65.000 | 56.750 | 60.474 | 0.833 | 0.208 | 9.167 | 9.375 | 9.083 |
| oof | R_channel_seq | 20 | 0 | N/A | 76.250 | 76.562 | 76.562 | 76.562 | N/A | N/A | 0.313 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_channel_seq | 20 | 2 | N/A | 76.250 | 76.562 | 77.500 | 75.521 | 51.875 | 60.657 | 0.313 | -0.938 | 1.979 | 1.042 | 23.646 |
| oof | R_channel_seq | 20 | 5 | N/A | 76.250 | 76.562 | 76.667 | 70.625 | 61.375 | 65.356 | 0.313 | -0.104 | 6.042 | 5.938 | 9.250 |
| oof | R_channel_seq | 20 | 10 | N/A | 76.250 | 76.562 | 76.667 | 68.542 | 64.063 | 66.184 | 0.313 | -0.104 | 8.125 | 8.021 | 4.479 |
| oof | R_channel_seq | 20 | 20 | N/A | 76.250 | 76.562 | 77.500 | 66.354 | 63.844 | 65.030 | 0.313 | -0.938 | 11.146 | 10.208 | 3.552 |
| proxy | R0 | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 5 | 0 | N/A | 50.938 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R0 | 5 | 2 | N/A | 50.938 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | 0.000 | -0.729 | 2.396 | 1.667 | 27.187 |
| proxy | R0 | 5 | 5 | N/A | 50.938 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | 0.000 | -0.625 | 5.833 | 5.208 | 13.437 |
| proxy | R0 | 5 | 10 | N/A | 50.938 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | 0.000 | -0.208 | 8.854 | 8.646 | 10.312 |
| proxy | R0 | 5 | 20 | N/A | 50.938 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | 0.000 | -0.833 | 13.333 | 12.500 | 12.281 |
| proxy | R0 | 10 | 0 | N/A | 53.032 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R0 | 10 | 2 | N/A | 53.032 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | 0.000 | -1.019 | 3.032 | 2.014 | 29.537 |
| proxy | R0 | 10 | 5 | N/A | 53.032 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | 0.000 | -1.412 | 7.245 | 5.833 | 14.199 |
| proxy | R0 | 10 | 10 | N/A | 53.032 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | 0.000 | -1.319 | 10.579 | 9.259 | 11.741 |
| proxy | R0 | 10 | 20 | N/A | 53.032 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | 0.000 | -1.481 | 14.815 | 13.333 | 13.285 |
| proxy | R0 | 20 | 0 | N/A | 52.473 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R0 | 20 | 2 | N/A | 52.473 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | 0.000 | -0.137 | 2.966 | 2.829 | 25.323 |
| proxy | R0 | 20 | 5 | N/A | 52.473 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | 0.000 | -0.466 | 7.456 | 6.990 | 12.357 |
| proxy | R0 | 20 | 10 | N/A | 52.473 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | 0.000 | -0.444 | 11.283 | 10.839 | 10.398 |
| proxy | R0 | 20 | 20 | N/A | 52.473 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | 0.000 | -0.948 | 15.351 | 14.402 | 10.174 |
| proxy | R_channel_reset | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_reset | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_reset | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_reset | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_reset | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_reset | 5 | 0 | N/A | 50.938 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_channel_reset | 5 | 2 | N/A | 50.938 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | 0.000 | -0.729 | 2.396 | 1.667 | 27.187 |
| proxy | R_channel_reset | 5 | 5 | N/A | 50.938 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | 0.000 | -0.625 | 5.833 | 5.208 | 13.437 |
| proxy | R_channel_reset | 5 | 10 | N/A | 50.938 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | 0.000 | -0.208 | 8.854 | 8.646 | 10.312 |
| proxy | R_channel_reset | 5 | 20 | N/A | 50.938 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | 0.000 | -0.833 | 13.333 | 12.500 | 12.281 |
| proxy | R_channel_reset | 10 | 0 | N/A | 53.032 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_channel_reset | 10 | 2 | N/A | 53.032 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | 0.000 | -1.019 | 3.032 | 2.014 | 29.537 |
| proxy | R_channel_reset | 10 | 5 | N/A | 53.032 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | 0.000 | -1.412 | 7.245 | 5.833 | 14.199 |
| proxy | R_channel_reset | 10 | 10 | N/A | 53.032 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | 0.000 | -1.319 | 10.579 | 9.259 | 11.741 |
| proxy | R_channel_reset | 10 | 20 | N/A | 53.032 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | 0.000 | -1.481 | 14.815 | 13.333 | 13.285 |
| proxy | R_channel_reset | 20 | 0 | N/A | 52.473 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_channel_reset | 20 | 2 | N/A | 52.473 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | 0.000 | -0.137 | 2.966 | 2.829 | 25.323 |
| proxy | R_channel_reset | 20 | 5 | N/A | 52.473 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | 0.000 | -0.466 | 7.456 | 6.990 | 12.357 |
| proxy | R_channel_reset | 20 | 10 | N/A | 52.473 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | 0.000 | -0.444 | 11.283 | 10.839 | 10.398 |
| proxy | R_channel_reset | 20 | 20 | N/A | 52.473 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | 0.000 | -0.948 | 15.351 | 14.402 | 10.174 |
| proxy | R_channel_seq | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_seq | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_seq | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_seq | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_seq | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_channel_seq | 5 | 0 | N/A | 50.938 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_channel_seq | 5 | 2 | N/A | 50.938 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | 0.000 | -0.729 | 2.396 | 1.667 | 27.187 |
| proxy | R_channel_seq | 5 | 5 | N/A | 50.938 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | 0.000 | -0.625 | 5.833 | 5.208 | 13.437 |
| proxy | R_channel_seq | 5 | 10 | N/A | 50.938 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | 0.000 | -0.208 | 8.854 | 8.646 | 10.312 |
| proxy | R_channel_seq | 5 | 20 | N/A | 50.938 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | 0.000 | -0.833 | 13.333 | 12.500 | 12.281 |
| proxy | R_channel_seq | 10 | 0 | N/A | 53.032 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_channel_seq | 10 | 2 | N/A | 53.032 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | 0.000 | -1.019 | 3.032 | 2.014 | 29.537 |
| proxy | R_channel_seq | 10 | 5 | N/A | 53.032 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | 0.000 | -1.412 | 7.245 | 5.833 | 14.199 |
| proxy | R_channel_seq | 10 | 10 | N/A | 53.032 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | 0.000 | -1.319 | 10.579 | 9.259 | 11.741 |
| proxy | R_channel_seq | 10 | 20 | N/A | 53.032 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | 0.000 | -1.481 | 14.815 | 13.333 | 13.285 |
| proxy | R_channel_seq | 20 | 0 | N/A | 52.473 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_channel_seq | 20 | 2 | N/A | 52.473 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | 0.000 | -0.137 | 2.966 | 2.829 | 25.323 |
| proxy | R_channel_seq | 20 | 5 | N/A | 52.473 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | 0.000 | -0.466 | 7.456 | 6.990 | 12.357 |
| proxy | R_channel_seq | 20 | 10 | N/A | 52.473 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | 0.000 | -0.444 | 11.283 | 10.839 | 10.398 |
| proxy | R_channel_seq | 20 | 20 | N/A | 52.473 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | 0.000 | -0.948 | 15.351 | 14.402 | 10.174 |

准确率为百分数，差值为百分点。全部分层、相对R0变化与六类正确性转换见CSV。内部训练目标、参数、梯度和完整8步轨迹见training_objectives，不把内部训练准确率称为独立验证。

实际有效训练阶段936个、投影更新7488次、内层head拟合25272次、新增最终head拟合936次。
实测运行墙钟2555.741 s；完整适配器和分类头状态、线程与分项工作量见summary.json。

- R0 remains original LocalRidge. R_channel_seq is the declared sequential channel-adapter candidate; R_channel_reset changes C initialization and proximal anchor to zero.
- Sequential and reset share one trained B and one C preparation; each inner head uses only its physical inner-train subset. C sequential inherits B adapter and anchor; C reset starts at zero.
- Only inner support squared-hinge margin loss supervises the channel adapter. Inner-held labels are part of training and are never called independent validation.
- Training uses eight projected Adam updates and a separate final objective, without best-step selection. Only physical K1/single-class stages skip; ordinary zero-gradient stages still execute and report all eight updates.
- True K1 has no fitting or held score. Proxy trainK1 uses exact R0 identity forward for every path; no K1 benefit is claimed.
- A is unavailable; B minus B0 is a support-classifier increment, not B minus ground A.
- OOF pools each physical held row once. Proxy averages all anchors within parent then weights parents equally.
- Old-only reuse and repeated old support across new-count rows are not independent observations.
- 736 stored channel parameters do not imply low total compute or a 5888-byte deployed model; full binding/head state, preparation memory and repeated inner solves are counted.
- Every original parameter/gradient coordinate is verified in the full raw streams. Summary curves retain all eight steps with per-block vector summaries; full vectors remain in the referenced immutable row logs.
- 10/1/3 percentage-point ideal directions are descriptive and impose no automatic promotion gate.
