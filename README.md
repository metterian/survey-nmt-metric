<p align="center">
  <img src="https://awesome.re/badge.svg" alt="Awesome">
  <img src="https://img.shields.io/badge/MDPI%20Mathematics-2023-2D5F8A?style=flat-square" alt="MDPI Mathematics 2023">
  <img src="https://img.shields.io/github/stars/metterian/survey-nmt-metric?style=flat-square" alt="Stars">
  <img src="https://img.shields.io/github/forks/metterian/survey-nmt-metric?style=flat-square" alt="Forks">
  <img src="https://img.shields.io/github/license/metterian/survey-nmt-metric?style=flat-square" alt="License">
</p>

<h1 align="center">A Survey on Evaluation Metrics for Machine Translation</h1>

<p align="center">
  <b>A curated list of papers on automatic evaluation metrics for machine translation</b>
</p>

<table align="center">
  <tr>
    <td align="center">
      <a href="https://scholar.google.com/citations?view_op=view_citation&citation_for_view=ajKxvvoAAAAJ:WF5omc3nYNoC">
        <img src="https://img.shields.io/badge/Google%20Scholar-133%2B%20Citations-FBBC04?style=for-the-badge&logo=google-scholar&logoColor=white" alt="133+ Citations">
      </a>
      <br>
      <sub><b>Highly Cited Survey Paper</b></sub>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://www.mdpi.com/2227-7390/11/4/1006"><img src="https://img.shields.io/badge/Paper-PDF-green?style=flat-square" alt="Paper"></a>
  <a href="https://youtu.be/9yvgzqPtKA4"><img src="https://img.shields.io/badge/Video-YouTube-red?style=flat-square&logo=youtube" alt="Video"></a>
  <a href="./ppt.pdf"><img src="https://img.shields.io/badge/Slides-PPT-orange?style=flat-square" alt="PPT"></a>
</p>

---

This repository accompanies our survey paper:

> **A Survey on Evaluation Metrics for Machine Translation**
> *Seungjun Lee, Jungseob Lee, Hyeonseok Moon, Chanjun Park, Jaehyung Seo, Sugyeong Eo, Seonmin Koo, Heuiseok Lim*
> Mathematics 2023, 11(4), 1006 [[Paper]](https://www.mdpi.com/2227-7390/11/4/1006)

We provide a comprehensive survey on automatic evaluation metrics for machine translation (MT), covering trends, taxonomy, key contributions, and shortcomings.

---

## Taxonomy

<p align="center">
  <img src="./metrics_tree.png" alt="Taxonomy of MT Evaluation Metrics" width="900">
</p>

---

## Table of Contents

- [1. Untrained Metrics](#1-untrained-metrics)
  - [1.1 Word-Based (N-gram)](#11-word-based-n-gram)
  - [1.2 Word-Based (Edit Distance)](#12-word-based-edit-distance)
  - [1.3 Character-Based](#13-character-based)
- [2. Trained Metrics](#2-trained-metrics)
  - [2.1 Static Embedding](#21-static-embedding)
  - [2.2 Contextualized Embedding](#22-contextualized-embedding)
- [3. WMT Metrics Shared Task](#3-wmt-metrics-shared-task)
- [4. Other Related Surveys](#4-other-related-surveys)

---

## 1. Untrained Metrics

### 1.1 Word-Based (N-gram)

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **BLEU** | [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) | 2002 | ACL |
| **NIST** | [Automatic Evaluation of Machine Translation Quality Using N-gram Co-Occurrence Statistics](https://dl.acm.org/doi/10.5555/1289189.1289273) | 2002 | HLT |
| **METEOR** | [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909/) | 2005 | ACL Workshop |
| **GTM** | [GTM: A Generic Translation Quality Metric](https://aclanthology.org/C04-1017/) | 2004 | COLING |

### 1.2 Word-Based (Edit Distance)

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **TER** | [A Study of Translation Edit Rate with Targeted Human Annotation](https://aclanthology.org/2006.amta-papers.25/) | 2006 | AMTA |
| **WER** | [A Comparison of Several Approximate Algorithms for Finding Multiple (N-best) Sentence Hypotheses](https://ieeexplore.ieee.org/document/225950) | 1994 | ICSLP |

### 1.3 Character-Based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **chrF** | [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) | 2015 | WMT |
| **chrF++** | [chrF++: words helping character n-grams](https://aclanthology.org/W17-4770/) | 2017 | WMT |

---

## 2. Trained Metrics

### 2.1 Static Embedding

#### Matching-based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **YiSi** | [YiSi - a Unified Semantic MT Quality Evaluation and Estimation Metric for Languages with Different Levels of Available Resources](https://aclanthology.org/W19-5358/) | 2019 | WMT |
| **MEANT** | [MEANT: An inexpensive, high-accuracy, semi-automatic metric for evaluating translation utility based on semantic roles](https://aclanthology.org/P11-1023/) | 2011 | ACL |
| **MoverScore** | [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://aclanthology.org/D19-1053/) | 2019 | EMNLP |

#### Regression-based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **BEER** | [BEER: BEtter Evaluation as Ranking](https://aclanthology.org/W14-3354/) | 2014 | WMT |
| **RUSE** | [RUSE: Regressor Using Sentence Embeddings for Automatic Machine Translation Evaluation](https://aclanthology.org/W18-6456/) | 2018 | WMT |
| **BLEND** | [BLEND: a Novel Combined MT Metric Based on Direct Assessment](https://aclanthology.org/W17-4768/) | 2017 | WMT |

### 2.2 Contextualized Embedding

#### Matching-based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **BERTScore** | [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) | 2020 | ICLR |

#### Regression-based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **BLEURT** | [BLEURT: Learning Robust Metrics for Text Generation](https://aclanthology.org/2020.acl-main.704/) | 2020 | ACL |
| **NUBIA** | [NUBIA: NeUral Based Interchangeability Assessor for Text Generation](https://arxiv.org/abs/2004.14667) | 2020 | arXiv |

#### Ranking-based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **COMET** | [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213/) | 2020 | EMNLP |
| **UniTE** | [UniTE: Unified Translation Evaluation](https://aclanthology.org/2022.acl-long.558/) | 2022 | ACL |

#### Generation-based

| Metric | Paper | Year | Venue |
|--------|-------|------|-------|
| **BARTScore** | [BARTScore: Evaluating Generated Text as Text Generation](https://arxiv.org/abs/2106.11520) | 2021 | NeurIPS |
| **Prism** | [Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing](https://aclanthology.org/2020.emnlp-main.8/) | 2020 | EMNLP |

---

## 3. WMT Metrics Shared Task

| Paper | Year | Venue |
|-------|------|-------|
| [Results of the WMT14 Metrics Shared Task](https://aclanthology.org/W14-3336/) | 2014 | WMT |
| [Results of the WMT15 Metrics Shared Task](https://aclanthology.org/W15-3031/) | 2015 | WMT |
| [Results of the WMT16 Metrics Shared Task](https://aclanthology.org/W16-2302/) | 2016 | WMT |
| [Results of the WMT17 Metrics Shared Task](https://aclanthology.org/W17-4755/) | 2017 | WMT |
| [Results of the WMT18 Metrics Shared Task](https://aclanthology.org/W18-6450/) | 2018 | WMT |
| [Results of the WMT19 Metrics Shared Task](https://aclanthology.org/W19-5302/) | 2019 | WMT |
| [Results of the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.77/) | 2020 | WMT |
| [Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73/) | 2021 | WMT |
| [Results of the WMT22 Metrics Shared Task](https://aclanthology.org/2022.wmt-1.2/) | 2022 | WMT |

---

## 4. Other Related Surveys

| Paper | Year | Venue |
|-------|------|-------|
| [A Survey of Evaluation Metrics Used for NLG Systems](https://dl.acm.org/doi/10.1145/3485766) | 2022 | ACM Computing Surveys |
| [A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios](https://aclanthology.org/N19-1423/) | 2019 | NAACL |

---

## Citation

If you find this survey useful for your research, please cite our paper:

```bibtex
@article{lee2023survey,
  title={A Survey on Evaluation Metrics for Machine Translation},
  author={Lee, Seungjun and Lee, Jungseob and Moon, Hyeonseok and Park, Chanjun and Seo, Jaehyung and Eo, Sugyeong and Koo, Seonmin and Lim, Heuiseok},
  journal={Mathematics},
  volume={11},
  number={4},
  pages={1006},
  year={2023},
  publisher={MDPI}
}
```

---

## Contributing

We welcome contributions! If you'd like to add a paper or fix an error, please open an issue or submit a pull request.

---

<p align="center">
  <sub>Maintained by <a href="https://github.com/metterian">@metterian</a></sub>
</p>
