# Enhancing Depression Detection from Narrative Interviews Using Language Models

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

- This repository is a tensorflow implementation of our work for the Depression Detection task proposed in our BIBM'23 paper:
[Enhancing Depression Detection from Narrative Interviews Using Language Models]

- In this work, we are also releasing the I-DAIC dataset for Depression Detection tasks in the healthcare domain. The I-DAIC consists of three datasets:
  - DAIC WOZ data
  - Extended DAIC WOZ data
  - EATD data
   
- You can apply for the DAIC WOZ and Extended DAIC WOZ access on [DAIC WOZ HOME](https://dcapswoz.ict.usc.edu/) The translated EATD corpus and originally translated corpus is provided in folder [EATD Dataset](https://github.com/palak97/IDAIC/tree/master/eatd_data) in this repository.

- - Links related to this work:
  - Dataset and codes: https://github.com/wangpinggl/IDAIC
  - Slides: https://github.com/wangpinggl/TREQS/blob/master/TREQS.pdf
  - Presentation Video: [Video](https://drive.google.com/open?id=1tXRaobsz1BWUJpzV976pgox_46c8jkPE](https://drive.google.com/file/d/1HwG_eiURv7OLgFvlNJNR4xVAFIi8t4Ud/view?usp=sharing))

## Citation
Sood, Palak, Xinming Yang, and Ping Wang. “Enhancing Depression Detection from Narrative Interviews Using Language Models.” In 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. 3173-3180. IEEE, 2023.

```
@inproceedings{sood2023enhancing,
  title={Enhancing Depression Detection from Narrative Interviews Using Language Models},
  author={Sood, Palak and Yang, Xinming and Wang, Ping},
  booktitle={2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3173--3180},
  year={2023},
  organization={IEEE}
}
```

## Results

Here we provide the results of different models presented below. Performance metrics for each class (depression and non-depression) were evaluated separately. The weighted evaluation accounts for label imbalance in the dataset.

<table style="width:100%">
  <colgroup>
    <col span="1" style="width: 10%;">
    <col span="3" style="width: 30%;">
    <col span="3" style="width: 30%;">
    <col span="3" style="width: 20%;">
    <col span="1" style="width: 10%;">
  </colgroup>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Non-Depression</th>
    <th colspan="3">Depression</th>
    <th colspan="3">Weighted</th>
    <th rowspan="2">KL Divergence</th>
  </tr>
  <tr>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>SVM</td>
    <td style="text-decoration: underline;">0.84</td>
    <td>0.94</td>
    <td style="text-decoration: underline;">0.89</td>
    <td>0.79</td>
    <td style="text-decoration: underline;">0.56</td>
    <td style="text-decoration: underline;">0.66</td>
    <td style="text-decoration: underline;">0.83</td>
    <td style="text-decoration: underline;">0.83</td>
    <td style="text-decoration: underline;">0.82</td>
    <td>0.0357</td>
  </tr>
    <tr>
    <td>LR</td>
    <td>0.82</td>
    <td style="text-decoration: underline;">0.96</td>
    <td style="text-decoration: underline;">0.89</td>
    <td style="text-decoration: underline;">0.84</td>
    <td>0.47</td>
    <td>0.60</td>
    <td style="text-decoration: underline;">0.83</td>
    <td>0.82</td>
    <td>0.81</td>
    <td style="text-decoration: underline;">0.0324</td>
  </tr>
  <tr>
    <td colspan="11" style="background-color: #d9d9d9;">BERT Models</td>
  </tr>
  <tr>
    <td>BERT$_{Base}$</td>
    <td>0.75</td>
    <td>0.68</td>
    <td>0.72</td>
    <td>0.36</td>
    <td>0.44</td>
    <td>0.39</td>
    <td>0.64</td>
    <td>0.61</td>
    <td>0.62</td>
    <td>0.0526</td>
  </tr>
  <tr>
    <td>BERT$_{Trunc}$</td>
    <td>0.79</td>
    <td>0.81</td>
    <td>0.80</td>
    <td>0.50</td>
    <td>0.47</td>
    <td>0.48</td>
    <td>0.71</td>
    <td>0.71</td>
    <td>0.71</td>
    <td>0.0461</td>
  </tr>
  <tr>
    <td>BERT$_{Window}$</td>
    <td>0.78</td>
    <td>0.88</td>
    <td style="text-decoration: underline;">0.83</td>
    <td>0.57</td>
    <td>0.38</td>
    <td>0.46</td>
    <td>0.72</td>
    <td>0.74</td>
    <td>0.72</td>
    <td>0.0417</td>
  </tr>
  <tr>
    <td>BERT$_{Sum\_wc}$</td>
    <td>0.75</td>
    <td style="font-weight: bold;">0.98</td>
    <td style="font-weight: bold;">0.85</td>
    <td style="font-weight: bold;">0.75</td>
    <td>0.18</td>
    <td>0.29</td>
    <td>0.75</td>
    <td>0.75</td>
    <td>0.69</td>
    <td>0.0441</td>
  </tr>
  <tr>
    <td>BERT$_{Sum\_wf}$</td>
    <td>0.73</td>
    <td>0.85</td>
    <td>0.79</td>
    <td>0.38</td>
    <td>0.24</td>
    <td>0.29</td>
    <td>0.63</td>
    <td>0.67</td>
    <td>0.65</td>
    <td>0.0466</td>
  </tr> 
    <tr>
    <td colspan="11" style="background-color: #d9d9d9;">RoBERTa Models</td>
  </tr>
  <tr>
    <td>RoBERTa$_{Base}$</td>
    <td>0.73</td>
    <td>0.82</td>
    <td>0.77</td>
    <td>0.35</td>
    <td>0.24</td>
    <td>0.28</td>
    <td>0.62</td>
    <td>0.66</td>
    <td>0.63</td>
    <td>0.0545</td>
  </tr>
  <tr>
    <td>RoBERTa$_{Trunc}$</td>
    <td style="font-weight: bold;">0.89</td>
    <td>0.56</td>
    <td>0.69</td>
    <td>0.43</td>
    <td style="font-weight: bold;">0.82</td>
    <td>0.57</td>
    <td>0.76</td>
    <td>0.64</td>
    <td>0.65</td>
    <td>0.0503</td>
  </tr>
  <tr>
    <td>RoBERTa$_{Window}$</td>
    <td>0.75</td>
    <td style="text-decoration: underline;">0.94</td>
    <td style="text-decoration: underline;">0.83</td>
    <td>0.58</td>
    <td>0.21</td>
    <td>0.30</td>
    <td>0.70</td>
    <td>0.73</td>
    <td>0.68</td>
    <td style="font-weight: bold;">0.0352</td>
  </tr>
  <tr>
    <td>RoBERTa$_{Sum\_wc}$</td>
    <td style="font-weight: bold;">0.89</td>
    <td>0.76</td>
    <td>0.82</td>
    <td>0.57</td>
    <td style="font-weight: bold;">0.76</td>
    <td style="font-weight: bold;">0.65</td>
    <td style="font-weight: bold;">0.80</td>
    <td style="font-weight: bold;">0.76</td>
    <td style="font-weight: bold;">0.77</td>
    <td>0.0390</td>
  </tr>
  <tr>
    <td>RoBERTa$_{Sum\_wf}$</td>
    <td>0.85</td>
    <td>0.85</td>
    <td style="font-weight: bold;">0.85</td>
    <td>0.62</td>
    <td>0.62</td>
    <td style="font-weight: bold;">0.62</td>
    <td>0.78</td>
    <td style="font-weight: bold;">0.78</td>
    <td style="font-weight: bold;">0.78</td>
    <td>0.0376</td>
  </tr>
</table>

