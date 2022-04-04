# DisOptNet: Distilling Semantic Knowledge from Optical Images for Weather-independent Building Segmentation

---

[Jian Kang](https://github.com/jiankang1991), Zhirui Wang, Zhirui Wang, [Junshi Xia](https://www.geoinformatics2018.com/member/junshixia/), [Xian Sun](http://people.ucas.ac.cn/~sunxian), [Ruben Fernandez-Beltran](https://scholar.google.es/citations?user=pdzJmcQAAAAJ&hl=es), [Antonio Plaza](https://www.umbc.edu/rssipl/people/aplaza/)

This repo contains the codes for the TGRS paper: [DisOptNet: Distilling Semantic Knowledge from Optical Images for Weather-independent Building Segmentation](). Compared to optical images, the semantics inherent to SAR images are less rich and interpretable due to factors like speckle noise and imaging geometry. In this scenario, most state-of-the-art methods are focused on designing advanced network architectures or loss functions for building footprint extraction. However, few works have been oriented towards improving segmentation performance through knowledge transfer from optical images. In this paper, we propose a novel method based on the DisOptNet network which can distill the useful semantic knowledge from optical images into a network only trained with SAR data. 


<p align="center">
<img src="pics/framework.png" alt="drawing"/>
</p>

## Usage

```
# first stage training (RGB)
./run_st1.sh
# second stage training (SAR+RGB)
./run_st2.sh
```

## Citation

```
@article{kang2021RiDe,
  title={DisOptNet: Distilling Semantic Knowledge from Optical Images for Weather-independent Building Segmentation},
  author={Kang, Jian and Wang, Zhirui and Zhu, Ruoxin and Xia, Junshi and Sun, Xian and Fernandez-Beltran, Ruben and Plaza, Antonio},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  note={DOI:10.1109/TGRS.2022.3165209}
  publisher={IEEE}
}
```


