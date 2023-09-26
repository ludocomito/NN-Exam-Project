
# NN-Exam-Project
<p>This repository hosts the notebook and all the needed information to replicate the Nerual Networks Exam project based upon the work presented in the seminal paper "Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks" by N. Wu et al. .<br>
  
The purpose of the study is to prove the probably counterintuitive hypothesis that in the context of multimodal learining methods one modality overcomes the other in a greedy manner, crippling the beneficial effects of said learning methods. The analysis begins with the implementation of an already well known project "MMTM: Multimodal Transfer Module for CNN Fusion" by H.R.V. Joze et al. to which a corrective method is applied to counter its greedy nature. The original MMTM and its proposed correction are run and compared through the use of custom defined metrics aimed at pointing out the distribution of the learning process over the various modalities of the multimodal learning process.<br>
More in depth the model uses two modalities and the proposed metric to evaluate their distribution is defined as the Conditional Utilisation Rate (CUR). The CUR is the relative change in accuracy between the two models within each pair. For example, u(m0|m1) measures the marginal contribution that m0 has in increasing the accuracy of the prediction of the fucntion of modality 1 and u(m1|m0) vice versa.<br>
  
The goal is to have the difference between the two CURs as low as possible. Since CURs are designed to be measured after training, a new metric is defined: Conditional Learning Speed (CLS). The conditional learning speed, s(m1|m0;t), is the log-ratio between the learning speed of the parameter from the fusion module and the original parameter of the uni-modal branch of modality 0. Same goes for s(m0|m1;t) for modality 1.<br>

The goal therefore becomes to have the minimal difference between CLSs, defined in the script as d_CLS.<br>

The chosen dataset is Modelnet40 amidst the ones proposed by the paper. More specifically 12 views for each render of the pointcloud files.
</p>

## Running the experiment

All the code needed to run training, evaluation and testing of the experiment is contained in a single notebook. In order to run the code correctly, follow these steps:

1. Create a main directory and place the notebook there
2. Download the ModelNet dataset [here](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz) (or follow the below commands), extract the dataset and place it in the main folder.
3. Create a logging and checkpoints folder.

You can replicate the above steps using these instructions:

```bash
mkdir greedy_multimodal_learning_main
cd greedy_multimodal_learning_main
curl -o ModelNet.tar.gz http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz
mkdir logging checkpoints
```

Following the best practice from the original paper, before running the full experiment create a .npy version of the classes in the dataset for efficiency. In order to do it, change the parameter *make_npy_files* to True in order to create the .npy files and execute the cells until the following one:

```python
if parameters_dict['make_npy_files']:
    _, _, _ = create_dataloaders(make_npy_files=parameters_dict['make_npy_files'])
```

Once finished, change *make_npy_files*to False and run the cells from the beginning.

Now you will be able to correctly run the notebook.
