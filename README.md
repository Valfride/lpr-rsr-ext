# Rodosol-LR-HR Dataset


This dataset contains 20,000 license plate (LP) images created from  [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset, which comprises 10,000 images of different cars models, captured during the day and ninght, from distinct pay tolls, on clear and rainy day.

Among the 10,000 original images, there are 5,000 images of mercorsur and brazilian LP layouts. While all Brazilian LPs consist of three letters followed by four digits, the initial pattern adopted in Brazil for Mercosur LPs consists of three letters, one digit, one letter and two digits, in that order (this is the pattern adopted on all Mercosur LPs in that dataset). In both layouts, car LPs have seven characters arranged in one row.

Here are some representative examples of the RodoSol-LR-HR dataset:
<img src="./Media/RodoSol_dataset.png" width="600"/>

# PKU-LR-HR Dataset

The PKU-LR-HR dataset comprises images categorized into five distinct groups, namely G1 through G5, each representing a specific scenario. For instance, the images in G1 were captured on highways during the day and depict a single vehicle. On the other hand, the images in G5 were taken at crosswalk intersections, either during the day or night, and have multiple vehicles. All images were collected in mainland China. Despite the diverse settings, the \gls*{lp} images have good quality and are perfectly legible.

Here are some representative examples of the PKU-LR-HR Dataset dataset:
<img src="./Media/PKU_dataset.png" width="600"/>

# LR-HR generation

The HR images used in our experiments were generated as follows. For each image from the chosen datasets, we first cropped the LP region using the annotations provided by the authors. Afterward, we used the same annotations to rectify each LP image so that it becomes more horizontal, tightly bounded, and easier to recognize. The rectified image is the HR image.

We generated LR versions of each HR image by simulating the effects of an optical system with lower resolution. This was achieved by iteratively applying random Gaussian noise to each HR image until we reached the desired degradation level for a given LR image (i.e., SSIM<0.1). To maintain the aspect ratio of the LR and HR images, we padded them before resizing.

<img src="./Media/image2.png" width="600"/>
Some HR-LR image pairs created from the RodoSol-ALPR dataset.

<img src="./Media/image.png" width="600"/>
Examples of HR-LR image pairs created from the PKU dataset.

