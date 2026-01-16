<p align="center">
  <img src="https://awesome.re/badge.svg" alt="Awesome">
  <img src="https://img.shields.io/github/stars/metterian/survey-nmt-metric?style=flat-square" alt="Stars">
  <img src="https://img.shields.io/github/forks/metterian/survey-nmt-metric?style=flat-square" alt="Forks">
  <img src="https://img.shields.io/github/license/metterian/survey-nmt-metric?style=flat-square" alt="License">
</p>

<h1 align="center">A Survey on Evaluation Metrics for Machine Translation</h1>

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

## Table of Contents

- [Survey Papers](#survey-papers)
- [Lexical-Based Metrics](#lexical-based-metrics)
  - [Word-Based](#word-based)
  - [Character-Based](#character-based)
- [Embedding-Based Metrics](#embedding-based-metrics)
  - [Word Embedding](#word-embedding)
  - [Contextual Embedding](#contextual-embedding)
- [Supervised Metrics](#supervised-metrics)
  - [Regression & Ranking](#regression--ranking)
- [Contributing](#contributing)
- [License](#license)

---

## Survey Papers

**A Survey on Evaluation Metrics for Machine Translation.** Seungjun Lee, Jungseob Lee, Hyeonseok Moon, Chanjun Park, Jaehyung Seo, Sugyeong Eo, Seonmin Koo, and Heuiseok Lim. *Mathematics*. 2023. [[paper]](https://www.mdpi.com/2227-7390/11/4/1006)

---

## Lexical-Based Metrics

Lexical-based metrics measure the similarity between the hypothesis and reference by comparing lexical items (words or phrases) without using deep learning algorithms.

### Word-Based

- **[BLEU]** Bleu: A method for automatic evaluation of machine translation. K. Papineni, S. Roukos, T. Ward, and W.J. Zhu. *ACL*. 2002. [[paper]](https://aclanthology.org/P02-1040/)

- **[NIST]** Automatic evaluation of machine translation quality using n-gram co-occurrence statistics. G. Doddington. *HLT*. 2002. [[paper]](https://dl.acm.org/doi/10.5555/1289189.1289273)

- **[WER]** An Information Theoretic Measure of Speech Recognition Performance. J. Woodard and J. Nelson. *IDIAP*. 1982. [[paper]](https://www.isca-archive.org/icassp_1982/woodard82_icassp.html)

- **[TER]** A study of translation edit rate with targeted human annotation. M. Snover, B. Dorr, R. Schwartz, L. Micciulla, and J. Makhoul. *AMTA*. 2006. [[paper]](https://aclanthology.org/2006.amta-papers.25/)

- **[GTM]** Evaluation of Machine Translation and Its Evaluation. J.P. Turian, L. Shea, and I.D. Melamed. *Technical Report*. 2006. [[paper]](https://aclanthology.org/C04-1017/)

- **[METEOR]** METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. S. Banerjee and A. Lavie. *ACL Workshop*. 2005. [[paper]](https://aclanthology.org/W05-0909/)

### Character-Based

- **[chrF]** chrF: Character n-gram F-score for automatic MT evaluation. M. Popović. *WMT*. 2015. [[paper]](https://aclanthology.org/W15-3049/)

---

## Embedding-Based Metrics

Embedding-based metrics measure similarity using machine learning or deep learning algorithms (dense vectors) to understand the word or context more deeply.

### Word Embedding

- **[MEANT]** MEANT: An inexpensive, high-accuracy, semi-automatic metric for evaluating translation utility based on semantic roles. C.k. Lo and D. Wu. *ACL*. 2011. [[paper]](https://aclanthology.org/P11-1023/)

- **[MEANT 2.0]** MEANT 2.0: Accurate semantic MT evaluation for any output language. C.k. Lo. *WMT*. 2017. [[paper]](https://aclanthology.org/W17-4767/)

### Contextual Embedding

- **[YiSi]** YiSi-a unified semantic MT quality evaluation and estimation metric for languages with different levels of available resources. C.k. Lo. *WMT*. 2019. [[paper]](https://aclanthology.org/W19-5358/)

- **[BERTscore]** Bertscore: Evaluating text generation with bert. T. Zhang, V. Kishore, F. Wu, K.Q. Weinberger, and Y. Artzi. *arXiv*. 2019. [[paper]](https://arxiv.org/abs/1904.09675)

- **[BARTscore]** Bartscore: Evaluating generated text as text generation. W. Yuan, G. Neubig, and P. Liu. *NeurIPS*. 2021. [[paper]](https://arxiv.org/abs/2106.11520)

---

## Supervised Metrics

Supervised metrics are trained using labeled data (usually human judgments) to predict quality scores.

### Regression & Ranking

- **[BEER]** Beer: Better evaluation as ranking. M. Stanojević and K. Sima'an. *WMT*. 2014. [[paper]](https://aclanthology.org/W14-3354/)

- **[BLEND]** Blend: A novel combined MT metric based on direct assessment. Q. Ma, Y. Graham, S. Wang, and Q. Liu. *WMT*. 2017. [[paper]](https://aclanthology.org/W17-4768/)

- **[RUSE]** Ruse: Regressor using sentence embeddings for automatic machine translation evaluation. H. Shimanaka, T. Kajiwara, and M. Komachi. *WMT*. 2018. [[paper]](https://aclanthology.org/W18-6456/)

- **[BERT for MTE]** Machine translation evaluation with bert regressor. H. Shimanaka, T. Kajiwara, and M. Komachi. *arXiv*. 2019. [[paper]](https://arxiv.org/abs/1907.12679)

- **[BLEURT]** BLEURT: Learning robust metrics for text generation. T. Sellam, D. Das, and A.P. Parikh. *arXiv*. 2020. [[paper]](https://arxiv.org/abs/2004.04696)

- **[NUBIA]** NUBIA: NeUral based interchangeability assessor for text generation. H. Kane, M.Y. Kocyigit, A. Abdalla, P. Ajanoh, and M. Coulibali. *arXiv*. 2020. [[paper]](https://arxiv.org/abs/2004.14667)

- **[COMET]** COMET: A neural framework for MT evaluation. R. Rei, C. Stewart, A.C. Farinha, and A. Lavie. *arXiv*. 2020. [[paper]](https://arxiv.org/abs/2009.09025)

---

## Contributing

Contributions are welcome.

---

## License

Distributed under the MIT License.

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
