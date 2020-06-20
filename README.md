# Modeling Product's Intrinsic and Extrinsic Characteristics for Recommender Systems

This is our official implementation for the paper:
> Bin Wu, Xiangnan He, Yun Chen, Liqiang Nie, Kai Zheng, and Yangdong Ye, Modeling Product's Intrinsic and Extrinsic Characteristics for Recommender Systems, IEEE TKDE, ACCEPT,2020.

If you use the codes, please cite our paper . Thanks!

## Model

The core model is `src/librec/ranking/IEPMF.java`.

## Dataset
We exhibit a small dataset `Office_Products.txt`, in which the columns indicate the `user_id`, `item_id`, `rating score`, respectively.
Please first unzip the file `Extrinsic_features_Office_Products.zip`.

## Quick Start

Run the main function in `src/librec/main/main.java`.

## Results

The outputs with default settings are as below:

```
[INFO ] 2020-01-03 23:36:49,630 -- With Setup: given-ratio -r 0.8 -target u --rand-seed 1 --test-view all
[DEBUG] 2020-01-03 23:36:49,733 -- training amount: 116068, test amount: 29073
[DEBUG] 2020-01-03 23:38:10,798 -- IEPMF: [ alpha, factors, regU, regI,beta,gama,lamutaE, numIters] = [10.0,10,0.1,0.1,5.0,500.0,10.0,100]
[DEBUG] 2020-01-03 23:38:10,801 -- IEPMF runs at iteration = 1 Fri Jan 03 23:38:10 CST 2020
[DEBUG] 2020-01-03 23:38:23,375 -- IEPMF runs at iteration = 2 Fri Jan 03 23:38:23 CST 2020
[DEBUG] 2020-01-03 23:38:52,133 -- IEPMF runs at iteration = 3 Fri Jan 03 23:38:52 CST 2020
[DEBUG] 2020-01-03 23:39:07,725 -- IEPMF runs at iteration = 4 Fri Jan 03 23:39:07 CST 2020
[DEBUG] 2020-01-03 23:39:23,432 -- IEPMF runs at iteration = 5 Fri Jan 03 23:39:23 CST 2020
[DEBUG] 2020-01-03 23:39:39,092 -- IEPMF runs at iteration = 6 Fri Jan 03 23:39:39 CST 2020
[DEBUG] 2020-01-03 23:39:54,685 -- IEPMF runs at iteration = 7 Fri Jan 03 23:39:54 CST 2020
[DEBUG] 2020-01-03 23:40:10,187 -- IEPMF runs at iteration = 8 Fri Jan 03 23:40:10 CST 2020
[DEBUG] 2020-01-03 23:40:25,885 -- IEPMF runs at iteration = 9 Fri Jan 03 23:40:25 CST 2020
[DEBUG] 2020-01-03 23:40:41,578 -- IEPMF runs at iteration = 10 Fri Jan 03 23:40:41 CST 2020
[DEBUG] 2020-01-03 23:40:56,666 -- IEPMF runs at iteration = 11 Fri Jan 03 23:40:56 CST 2020
[DEBUG] 2020-01-03 23:41:11,347 -- IEPMF runs at iteration = 12 Fri Jan 03 23:41:11 CST 2020
[DEBUG] 2020-01-03 23:41:26,704 -- IEPMF runs at iteration = 13 Fri Jan 03 23:41:26 CST 2020
[DEBUG] 2020-01-03 23:41:41,858 -- IEPMF runs at iteration = 14 Fri Jan 03 23:41:41 CST 2020
[DEBUG] 2020-01-03 23:41:57,075 -- IEPMF runs at iteration = 15 Fri Jan 03 23:41:57 CST 2020
[DEBUG] 2020-01-03 23:42:12,265 -- IEPMF runs at iteration = 16 Fri Jan 03 23:42:12 CST 2020
[DEBUG] 2020-01-03 23:42:27,430 -- IEPMF runs at iteration = 17 Fri Jan 03 23:42:27 CST 2020
[DEBUG] 2020-01-03 23:42:42,343 -- IEPMF runs at iteration = 18 Fri Jan 03 23:42:42 CST 2020
[DEBUG] 2020-01-03 23:42:57,494 -- IEPMF runs at iteration = 19 Fri Jan 03 23:42:57 CST 2020
[DEBUG] 2020-01-03 23:43:12,624 -- IEPMF runs at iteration = 20 Fri Jan 03 23:43:12 CST 2020
[DEBUG] 2020-01-03 23:43:27,765 -- IEPMF runs at iteration = 21 Fri Jan 03 23:43:27 CST 2020
[DEBUG] 2020-01-03 23:43:42,900 -- IEPMF runs at iteration = 22 Fri Jan 03 23:43:42 CST 2020
[DEBUG] 2020-01-03 23:43:58,058 -- IEPMF runs at iteration = 23 Fri Jan 03 23:43:58 CST 2020
[DEBUG] 2020-01-03 23:44:13,198 -- IEPMF runs at iteration = 24 Fri Jan 03 23:44:13 CST 2020
[DEBUG] 2020-01-03 23:44:28,335 -- IEPMF runs at iteration = 25 Fri Jan 03 23:44:28 CST 2020
[DEBUG] 2020-01-03 23:44:43,467 -- IEPMF runs at iteration = 26 Fri Jan 03 23:44:43 CST 2020
[DEBUG] 2020-01-03 23:44:58,633 -- IEPMF runs at iteration = 27 Fri Jan 03 23:44:58 CST 2020
[DEBUG] 2020-01-03 23:45:13,747 -- IEPMF runs at iteration = 28 Fri Jan 03 23:45:13 CST 2020
[DEBUG] 2020-01-03 23:45:28,575 -- IEPMF runs at iteration = 29 Fri Jan 03 23:45:28 CST 2020
[DEBUG] 2020-01-03 23:45:43,711 -- IEPMF runs at iteration = 30 Fri Jan 03 23:45:43 CST 2020
[DEBUG] 2020-01-03 23:45:58,841 -- IEPMF runs at iteration = 31 Fri Jan 03 23:45:58 CST 2020
[DEBUG] 2020-01-03 23:46:13,971 -- IEPMF runs at iteration = 32 Fri Jan 03 23:46:13 CST 2020
[DEBUG] 2020-01-03 23:46:29,126 -- IEPMF runs at iteration = 33 Fri Jan 03 23:46:29 CST 2020
[DEBUG] 2020-01-03 23:46:44,391 -- IEPMF runs at iteration = 34 Fri Jan 03 23:46:44 CST 2020
[DEBUG] 2020-01-03 23:46:59,508 -- IEPMF runs at iteration = 35 Fri Jan 03 23:46:59 CST 2020
[DEBUG] 2020-01-03 23:47:14,632 -- IEPMF runs at iteration = 36 Fri Jan 03 23:47:14 CST 2020
[DEBUG] 2020-01-03 23:47:29,821 -- IEPMF runs at iteration = 37 Fri Jan 03 23:47:29 CST 2020
[DEBUG] 2020-01-03 23:47:44,982 -- IEPMF runs at iteration = 38 Fri Jan 03 23:47:44 CST 2020
[DEBUG] 2020-01-03 23:48:00,169 -- IEPMF runs at iteration = 39 Fri Jan 03 23:48:00 CST 2020
[DEBUG] 2020-01-03 23:48:15,235 -- IEPMF runs at iteration = 40 Fri Jan 03 23:48:15 CST 2020
[DEBUG] 2020-01-03 23:48:30,369 -- IEPMF runs at iteration = 41 Fri Jan 03 23:48:30 CST 2020
[DEBUG] 2020-01-03 23:48:45,342 -- IEPMF runs at iteration = 42 Fri Jan 03 23:48:45 CST 2020
[DEBUG] 2020-01-03 23:49:00,402 -- IEPMF runs at iteration = 43 Fri Jan 03 23:49:00 CST 2020
[DEBUG] 2020-01-03 23:49:15,594 -- IEPMF runs at iteration = 44 Fri Jan 03 23:49:15 CST 2020
[DEBUG] 2020-01-03 23:49:30,664 -- IEPMF runs at iteration = 45 Fri Jan 03 23:49:30 CST 2020
[DEBUG] 2020-01-03 23:49:45,800 -- IEPMF runs at iteration = 46 Fri Jan 03 23:49:45 CST 2020
[DEBUG] 2020-01-03 23:50:00,761 -- IEPMF runs at iteration = 47 Fri Jan 03 23:50:00 CST 2020
[DEBUG] 2020-01-03 23:50:15,834 -- IEPMF runs at iteration = 48 Fri Jan 03 23:50:15 CST 2020
[DEBUG] 2020-01-03 23:50:30,817 -- IEPMF runs at iteration = 49 Fri Jan 03 23:50:30 CST 2020
[DEBUG] 2020-01-03 23:50:45,980 -- IEPMF runs at iteration = 50 Fri Jan 03 23:50:45 CST 2020
[DEBUG] 2020-01-03 23:51:01,012 -- IEPMF runs at iteration = 51 Fri Jan 03 23:51:01 CST 2020
[DEBUG] 2020-01-03 23:51:16,063 -- IEPMF runs at iteration = 52 Fri Jan 03 23:51:16 CST 2020
[DEBUG] 2020-01-03 23:51:31,045 -- IEPMF runs at iteration = 53 Fri Jan 03 23:51:31 CST 2020
[DEBUG] 2020-01-03 23:51:46,132 -- IEPMF runs at iteration = 54 Fri Jan 03 23:51:46 CST 2020
[DEBUG] 2020-01-03 23:52:01,461 -- IEPMF runs at iteration = 55 Fri Jan 03 23:52:01 CST 2020
[DEBUG] 2020-01-03 23:52:16,570 -- IEPMF runs at iteration = 56 Fri Jan 03 23:52:16 CST 2020
[DEBUG] 2020-01-03 23:52:31,718 -- IEPMF runs at iteration = 57 Fri Jan 03 23:52:31 CST 2020
[DEBUG] 2020-01-03 23:52:46,863 -- IEPMF runs at iteration = 58 Fri Jan 03 23:52:46 CST 2020
[DEBUG] 2020-01-03 23:53:01,509 -- IEPMF runs at iteration = 59 Fri Jan 03 23:53:01 CST 2020
[DEBUG] 2020-01-03 23:53:16,238 -- IEPMF runs at iteration = 60 Fri Jan 03 23:53:16 CST 2020
[DEBUG] 2020-01-03 23:53:31,366 -- IEPMF runs at iteration = 61 Fri Jan 03 23:53:31 CST 2020
[DEBUG] 2020-01-03 23:53:46,130 -- IEPMF runs at iteration = 62 Fri Jan 03 23:53:46 CST 2020
[DEBUG] 2020-01-03 23:54:01,245 -- IEPMF runs at iteration = 63 Fri Jan 03 23:54:01 CST 2020
[DEBUG] 2020-01-03 23:54:16,086 -- IEPMF runs at iteration = 64 Fri Jan 03 23:54:16 CST 2020
[DEBUG] 2020-01-03 23:54:31,219 -- IEPMF runs at iteration = 65 Fri Jan 03 23:54:31 CST 2020
[DEBUG] 2020-01-03 23:54:46,373 -- IEPMF runs at iteration = 66 Fri Jan 03 23:54:46 CST 2020
[DEBUG] 2020-01-03 23:55:01,517 -- IEPMF runs at iteration = 67 Fri Jan 03 23:55:01 CST 2020
[DEBUG] 2020-01-03 23:55:16,556 -- IEPMF runs at iteration = 68 Fri Jan 03 23:55:16 CST 2020
[DEBUG] 2020-01-03 23:55:31,335 -- IEPMF runs at iteration = 69 Fri Jan 03 23:55:31 CST 2020
[DEBUG] 2020-01-03 23:55:46,474 -- IEPMF runs at iteration = 70 Fri Jan 03 23:55:46 CST 2020
[DEBUG] 2020-01-03 23:56:01,622 -- IEPMF runs at iteration = 71 Fri Jan 03 23:56:01 CST 2020
[DEBUG] 2020-01-03 23:56:16,885 -- IEPMF runs at iteration = 72 Fri Jan 03 23:56:16 CST 2020
[DEBUG] 2020-01-03 23:56:32,059 -- IEPMF runs at iteration = 73 Fri Jan 03 23:56:32 CST 2020
[DEBUG] 2020-01-03 23:56:47,199 -- IEPMF runs at iteration = 74 Fri Jan 03 23:56:47 CST 2020
[DEBUG] 2020-01-03 23:57:01,779 -- IEPMF runs at iteration = 75 Fri Jan 03 23:57:01 CST 2020
[DEBUG] 2020-01-03 23:57:16,941 -- IEPMF runs at iteration = 76 Fri Jan 03 23:57:16 CST 2020
[DEBUG] 2020-01-03 23:57:32,082 -- IEPMF runs at iteration = 77 Fri Jan 03 23:57:32 CST 2020
[DEBUG] 2020-01-03 23:57:46,897 -- IEPMF runs at iteration = 78 Fri Jan 03 23:57:46 CST 2020
[DEBUG] 2020-01-03 23:58:01,921 -- IEPMF runs at iteration = 79 Fri Jan 03 23:58:01 CST 2020
[DEBUG] 2020-01-03 23:58:17,069 -- IEPMF runs at iteration = 80 Fri Jan 03 23:58:17 CST 2020
[DEBUG] 2020-01-03 23:58:32,175 -- IEPMF runs at iteration = 81 Fri Jan 03 23:58:32 CST 2020
[DEBUG] 2020-01-03 23:58:47,314 -- IEPMF runs at iteration = 82 Fri Jan 03 23:58:47 CST 2020
[DEBUG] 2020-01-03 23:59:02,434 -- IEPMF runs at iteration = 83 Fri Jan 03 23:59:02 CST 2020
[DEBUG] 2020-01-03 23:59:17,599 -- IEPMF runs at iteration = 84 Fri Jan 03 23:59:17 CST 2020
[DEBUG] 2020-01-03 23:59:32,766 -- IEPMF runs at iteration = 85 Fri Jan 03 23:59:32 CST 2020
[DEBUG] 2020-01-03 23:59:47,899 -- IEPMF runs at iteration = 86 Fri Jan 03 23:59:47 CST 2020
[DEBUG] 2020-01-04 00:00:03,037 -- IEPMF runs at iteration = 87 Sat Jan 04 00:00:03 CST 2020
[DEBUG] 2020-01-04 00:00:18,176 -- IEPMF runs at iteration = 88 Sat Jan 04 00:00:18 CST 2020
[DEBUG] 2020-01-04 00:00:33,468 -- IEPMF runs at iteration = 89 Sat Jan 04 00:00:33 CST 2020
[DEBUG] 2020-01-04 00:00:48,669 -- IEPMF runs at iteration = 90 Sat Jan 04 00:00:48 CST 2020
[DEBUG] 2020-01-04 00:01:03,792 -- IEPMF runs at iteration = 91 Sat Jan 04 00:01:03 CST 2020
[DEBUG] 2020-01-04 00:01:18,924 -- IEPMF runs at iteration = 92 Sat Jan 04 00:01:18 CST 2020
[DEBUG] 2020-01-04 00:01:34,061 -- IEPMF runs at iteration = 93 Sat Jan 04 00:01:34 CST 2020
[DEBUG] 2020-01-04 00:01:48,746 -- IEPMF runs at iteration = 94 Sat Jan 04 00:01:48 CST 2020
[DEBUG] 2020-01-04 00:02:03,828 -- IEPMF runs at iteration = 95 Sat Jan 04 00:02:03 CST 2020
[DEBUG] 2020-01-04 00:02:18,720 -- IEPMF runs at iteration = 96 Sat Jan 04 00:02:18 CST 2020
[DEBUG] 2020-01-04 00:02:33,877 -- IEPMF runs at iteration = 97 Sat Jan 04 00:02:33 CST 2020
[DEBUG] 2020-01-04 00:02:49,122 -- IEPMF runs at iteration = 98 Sat Jan 04 00:02:49 CST 2020
[DEBUG] 2020-01-04 00:03:03,989 -- IEPMF runs at iteration = 99 Sat Jan 04 00:03:03 CST 2020
[DEBUG] 2020-01-04 00:03:19,094 -- IEPMF runs at iteration = 100 Sat Jan 04 00:03:19 CST 2020
[DEBUG] 2020-01-04 00:03:34,023 -- IEPMF evaluate test data ... 
[INFO ] 2020-01-04 00:03:46,047 -- IEPMF,Precision@10=0.009559,Recall@10=0.038752,MAP@10=0.014472,NDCG@10=0.024944,,10.0,10,0.1,0.1,5.0,500.0,10.0,100,'25:31','00:12'
