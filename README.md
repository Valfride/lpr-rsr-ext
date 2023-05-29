# Rodosol-LR-HR Dataset


The Rodosol-LR-HR dataset consists of 20,000 license plate (LP) images created from the [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset. The RodoSol-ALPR dataset comprises 10,000 images of different car models captured during both day and night at distinct toll booths, under clear and rainy weather conditions.

Among the 10,000 original images, 5,000 images feature Mercosur and Brazilian LP layouts. Brazilian LPs consist of three letters followed by four digits, while the initial pattern adopted in Brazil for Mercosur LPs consists of three letters, one digit, one letter, and two digits, in that order (this pattern is adopted for all Mercosur LPs in the dataset). In both layouts, car LPs have seven characters arranged in a single row.

Here are some representative examples of the Rodosol-LR-HR dataset:
<img src="./Media/RodoSol_dataset.png" width="600"/>

# PKU-LR-HR Dataset

The PKU-LR-HR dataset comprises images categorized into five distinct groups, namely G1 through G5, each representing a specific scenario. For example, G1 images were captured on highways during the day and depict a single vehicle, while G5 images were taken at crosswalk intersections, either during the day or night, and feature multiple vehicles. All images in this dataset were collected in mainland China. Despite the diverse settings, the LP images exhibit good quality and perfect legibility.

Here are some representative examples of the PKU-LR-HR dataset:
<img src="./Media/PKU_dataset.png" width="600"/>

# LR-HR generation

The HR images used in our experiments were generated as follows. For each image from the chosen datasets, we first cropped the LP region using the annotations provided by the authors. We then used the same annotations to rectify each LP image, making it more horizontal, tightly bounded, and easier to recognize. The rectified image serves as the HR image.

To create LR versions of each HR image, we simulated the effects of an optical system with lower resolution. This was achieved by iteratively applying random Gaussian noise to each HR image until we reached the desired degradation level for a given LR image (i.e., SSIM < 0.1). To maintain the aspect ratio of the LR and HR images, we padded them before resizing.

Here are some HR-LR image pairs created from the RodoSol-ALPR dataset:
<img src="./Media/image2.png" width="600"/>


And here are some examples of HR-LR image pairs created from the PKU dataset:
<img src="./Media/image.png" width="600"/>

