# FCR8-LocalRidge support pilot

完整160 parent、4 row、三路径均已核验；旧类固定6个。A与B−A为N/A。R_FCR8_seq为预声明顺序主线，R_FCR8_reset_init仅作继承对照。

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
| oof | R_FCR8_reset_init | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_reset_init | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_reset_init | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_reset_init | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_reset_init | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_reset_init | 5 | 0 | N/A | 63.333 | 62.917 | 62.917 | 62.917 | N/A | N/A | -0.417 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_FCR8_reset_init | 5 | 2 | N/A | 63.333 | 62.917 | 62.500 | 60.417 | 51.250 | 52.795 | -0.417 | 0.417 | 2.083 | 2.500 | 20.833 |
| oof | R_FCR8_reset_init | 5 | 5 | N/A | 63.333 | 62.917 | 62.500 | 55.417 | 48.500 | 51.311 | -0.417 | 0.417 | 7.083 | 7.500 | 8.417 |
| oof | R_FCR8_reset_init | 5 | 10 | N/A | 63.333 | 62.917 | 64.167 | 52.500 | 48.000 | 49.804 | -0.417 | -1.250 | 11.667 | 10.417 | 8.167 |
| oof | R_FCR8_reset_init | 5 | 20 | N/A | 63.333 | 62.917 | 63.750 | 48.333 | 46.500 | 46.619 | -0.417 | -0.833 | 15.417 | 14.583 | 10.083 |
| oof | R_FCR8_reset_init | 10 | 0 | N/A | 73.542 | 73.333 | 73.333 | 73.333 | N/A | N/A | -0.208 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_FCR8_reset_init | 10 | 2 | N/A | 73.542 | 73.333 | 72.708 | 70.833 | 53.750 | 59.886 | -0.208 | 0.625 | 1.875 | 2.500 | 21.250 |
| oof | R_FCR8_reset_init | 10 | 5 | N/A | 73.542 | 73.333 | 72.917 | 67.292 | 56.500 | 61.217 | -0.208 | 0.417 | 5.625 | 6.042 | 10.792 |
| oof | R_FCR8_reset_init | 10 | 10 | N/A | 73.542 | 73.333 | 73.125 | 66.250 | 58.000 | 61.830 | -0.208 | 0.208 | 6.875 | 7.083 | 8.250 |
| oof | R_FCR8_reset_init | 10 | 20 | N/A | 73.542 | 73.333 | 73.333 | 64.167 | 56.125 | 59.783 | -0.208 | 0.000 | 9.167 | 9.167 | 8.458 |
| oof | R_FCR8_reset_init | 20 | 0 | N/A | 76.250 | 76.354 | 76.354 | 76.354 | N/A | N/A | 0.104 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_FCR8_reset_init | 20 | 2 | N/A | 76.250 | 76.354 | 76.771 | 74.479 | 55.312 | 63.063 | 0.104 | -0.417 | 2.292 | 1.875 | 19.167 |
| oof | R_FCR8_reset_init | 20 | 5 | N/A | 76.250 | 76.354 | 76.562 | 70.625 | 61.375 | 65.342 | 0.104 | -0.208 | 5.938 | 5.729 | 9.458 |
| oof | R_FCR8_reset_init | 20 | 10 | N/A | 76.250 | 76.354 | 77.188 | 68.854 | 63.938 | 66.247 | 0.104 | -0.833 | 8.333 | 7.500 | 4.917 |
| oof | R_FCR8_reset_init | 20 | 20 | N/A | 76.250 | 76.354 | 76.979 | 65.833 | 62.969 | 64.318 | 0.104 | -0.625 | 11.146 | 10.521 | 3.865 |
| oof | R_FCR8_seq | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_seq | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_seq | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_seq | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_seq | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_FCR8_seq | 5 | 0 | N/A | 63.333 | 62.917 | 62.917 | 62.917 | N/A | N/A | -0.417 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_FCR8_seq | 5 | 2 | N/A | 63.333 | 62.917 | 62.500 | 60.417 | 48.750 | 51.546 | -0.417 | 0.417 | 2.083 | 2.500 | 21.667 |
| oof | R_FCR8_seq | 5 | 5 | N/A | 63.333 | 62.917 | 63.333 | 56.250 | 48.500 | 51.701 | -0.417 | -0.417 | 7.083 | 6.667 | 8.417 |
| oof | R_FCR8_seq | 5 | 10 | N/A | 63.333 | 62.917 | 63.333 | 53.750 | 48.250 | 50.520 | -0.417 | -0.417 | 9.583 | 9.167 | 7.833 |
| oof | R_FCR8_seq | 5 | 20 | N/A | 63.333 | 62.917 | 65.000 | 49.167 | 46.625 | 46.952 | -0.417 | -2.083 | 15.833 | 13.750 | 9.958 |
| oof | R_FCR8_seq | 10 | 0 | N/A | 73.542 | 73.333 | 73.333 | 73.333 | N/A | N/A | -0.208 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_FCR8_seq | 10 | 2 | N/A | 73.542 | 73.333 | 72.917 | 71.042 | 53.125 | 59.571 | -0.208 | 0.417 | 1.875 | 2.292 | 22.083 |
| oof | R_FCR8_seq | 10 | 5 | N/A | 73.542 | 73.333 | 73.542 | 67.917 | 56.250 | 61.316 | -0.208 | -0.208 | 5.625 | 5.417 | 11.667 |
| oof | R_FCR8_seq | 10 | 10 | N/A | 73.542 | 73.333 | 72.917 | 66.250 | 58.000 | 61.809 | -0.208 | 0.417 | 6.667 | 7.083 | 8.250 |
| oof | R_FCR8_seq | 10 | 20 | N/A | 73.542 | 73.333 | 73.125 | 63.958 | 55.937 | 59.589 | -0.208 | 0.208 | 9.167 | 9.375 | 8.312 |
| oof | R_FCR8_seq | 20 | 0 | N/A | 76.250 | 76.354 | 76.354 | 76.354 | N/A | N/A | 0.104 | 0.000 | 0.000 | 0.000 | N/A |
| oof | R_FCR8_seq | 20 | 2 | N/A | 76.250 | 76.354 | 76.562 | 74.479 | 55.313 | 63.144 | 0.104 | -0.208 | 2.083 | 1.875 | 19.167 |
| oof | R_FCR8_seq | 20 | 5 | N/A | 76.250 | 76.354 | 76.562 | 70.521 | 61.250 | 65.258 | 0.104 | -0.208 | 6.042 | 5.833 | 9.479 |
| oof | R_FCR8_seq | 20 | 10 | N/A | 76.250 | 76.354 | 77.187 | 69.167 | 63.938 | 66.396 | 0.104 | -0.833 | 8.021 | 7.187 | 5.229 |
| oof | R_FCR8_seq | 20 | 20 | N/A | 76.250 | 76.354 | 77.083 | 66.146 | 63.062 | 64.538 | 0.104 | -0.729 | 10.938 | 10.208 | 3.458 |
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
| proxy | R_FCR8_reset_init | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_reset_init | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_reset_init | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_reset_init | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_reset_init | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_reset_init | 5 | 0 | N/A | 50.938 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_FCR8_reset_init | 5 | 2 | N/A | 50.938 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | 0.000 | -0.729 | 2.396 | 1.667 | 27.187 |
| proxy | R_FCR8_reset_init | 5 | 5 | N/A | 50.938 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | 0.000 | -0.625 | 5.833 | 5.208 | 13.437 |
| proxy | R_FCR8_reset_init | 5 | 10 | N/A | 50.938 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | 0.000 | -0.208 | 8.854 | 8.646 | 10.312 |
| proxy | R_FCR8_reset_init | 5 | 20 | N/A | 50.938 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | 0.000 | -0.833 | 13.333 | 12.500 | 12.281 |
| proxy | R_FCR8_reset_init | 10 | 0 | N/A | 53.032 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_FCR8_reset_init | 10 | 2 | N/A | 53.032 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | 0.000 | -1.019 | 3.032 | 2.014 | 29.537 |
| proxy | R_FCR8_reset_init | 10 | 5 | N/A | 53.032 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | 0.000 | -1.412 | 7.245 | 5.833 | 14.199 |
| proxy | R_FCR8_reset_init | 10 | 10 | N/A | 53.032 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | 0.000 | -1.319 | 10.579 | 9.259 | 11.741 |
| proxy | R_FCR8_reset_init | 10 | 20 | N/A | 53.032 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | 0.000 | -1.481 | 14.815 | 13.333 | 13.285 |
| proxy | R_FCR8_reset_init | 20 | 0 | N/A | 52.473 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_FCR8_reset_init | 20 | 2 | N/A | 52.473 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | 0.000 | -0.137 | 2.966 | 2.829 | 25.323 |
| proxy | R_FCR8_reset_init | 20 | 5 | N/A | 52.473 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | 0.000 | -0.466 | 7.456 | 6.990 | 12.357 |
| proxy | R_FCR8_reset_init | 20 | 10 | N/A | 52.473 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | 0.000 | -0.444 | 11.283 | 10.839 | 10.398 |
| proxy | R_FCR8_reset_init | 20 | 20 | N/A | 52.473 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | 0.000 | -0.948 | 15.351 | 14.402 | 10.174 |
| proxy | R_FCR8_seq | 1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_seq | 1 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_seq | 1 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_seq | 1 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_seq | 1 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_FCR8_seq | 5 | 0 | N/A | 50.938 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_FCR8_seq | 5 | 2 | N/A | 50.938 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | 0.000 | -0.729 | 2.396 | 1.667 | 27.187 |
| proxy | R_FCR8_seq | 5 | 5 | N/A | 50.938 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | 0.000 | -0.625 | 5.833 | 5.208 | 13.437 |
| proxy | R_FCR8_seq | 5 | 10 | N/A | 50.938 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | 0.000 | -0.208 | 8.854 | 8.646 | 10.312 |
| proxy | R_FCR8_seq | 5 | 20 | N/A | 50.938 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | 0.000 | -0.833 | 13.333 | 12.500 | 12.281 |
| proxy | R_FCR8_seq | 10 | 0 | N/A | 53.032 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_FCR8_seq | 10 | 2 | N/A | 53.032 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | 0.000 | -1.019 | 3.032 | 2.014 | 29.537 |
| proxy | R_FCR8_seq | 10 | 5 | N/A | 53.032 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | 0.000 | -1.412 | 7.245 | 5.833 | 14.199 |
| proxy | R_FCR8_seq | 10 | 10 | N/A | 53.032 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | 0.000 | -1.319 | 10.579 | 9.259 | 11.741 |
| proxy | R_FCR8_seq | 10 | 20 | N/A | 53.032 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | 0.000 | -1.481 | 14.815 | 13.333 | 13.285 |
| proxy | R_FCR8_seq | 20 | 0 | N/A | 52.473 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| proxy | R_FCR8_seq | 20 | 2 | N/A | 52.473 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | 0.000 | -0.137 | 2.966 | 2.829 | 25.323 |
| proxy | R_FCR8_seq | 20 | 5 | N/A | 52.473 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | 0.000 | -0.466 | 7.456 | 6.990 | 12.357 |
| proxy | R_FCR8_seq | 20 | 10 | N/A | 52.473 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | 0.000 | -0.444 | 11.283 | 10.839 | 10.398 |
| proxy | R_FCR8_seq | 20 | 20 | N/A | 52.473 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | 0.000 | -0.948 | 15.351 | 14.402 | 10.174 |

准确率为百分数，差值为百分点。全部分层、相对R0变化与六类正确性转换见CSV。内部训练目标、参数、梯度、全部回溯试探和停止原因见training_objectives，不把内部训练准确率称为独立验证。

实际有效训练阶段936个、函数坐标更新2673次、内层head拟合25740次、新增最终head拟合936次。
实测运行墙钟3551.958 s；完整适配器和分类头状态、线程与分项工作量见summary.json。

- R0 remains original LocalRidge. R_FCR8_seq is the fixed main candidate; R_FCR8_reset_init changes C initialization, proximal anchor and the resulting stage keep limit to its own initial state, while retaining the same B teacher.
- Sequential and reset share one trained B and one C preparation/teacher. Every teacher old head and student head uses its own physical inner-train. C sequential exactly inherits original B U and builds fresh W with Z=0; reset_init uses U=0. Fixed DCT V0 is shared and never trained.
- Task class RMS and previous-class teacher-deficit RMS pool physical losses across folds before reduction; both coefficients are one. Inner-held labels supervise function-coordinate Z/original U and are not independent validation.
- Four iterations and three trials are fixed. The normalized function-coordinate total gradient is projected into the keep halfspace without a parameter ball. Each actual trial must pass Armijo, total-objective nonincrease and real keep-risk checks. Every rejection and bounded stop is verified; final parameters use the last accepted cache.
- True K1 has no fitting or held score. Proxy trainK1 uses exact R0 identity forward for every path; no K1 benefit is claimed.
- A is unavailable; B minus B0 is a support-classifier increment, not B minus ground A.
- OOF pools each physical held row once. Proxy averages all anchors within parent then weights parents equally.
- Old-only reuse and repeated old support across new-count rows are not independent observations.
- At most 5888 U parameters are trainable (736 times actual latent rank); U and materialized fixed V0 together occupy 94208 bytes; this is not total model size or proof of lower training cost. Baseline, prepaid B initial heads, C teacher heads, student trial heads, both adjoint channels and full deployed state are separately counted.
- Every parameter/gradient/direction coordinate and teacher score/q is preserved in exclusive NPZ archives, referenced by complete native finite JSON and checked against exact inventories/content. Compact logs retain Z/U/W and gradient array summaries plus measured forward mechanisms; bulk archives remain original artifacts and are not duplicated in reports.
- 10/1/3 percentage-point ideal directions are descriptive and impose no automatic promotion gate.
