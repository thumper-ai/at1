
# akash-thumper-v1 (AT1) Model 


## Overview

The `akash-thumper-v1` (AT1) is a creative commons-based text-to-image generation model, marking a significant milestone as the first foundational model trained entirely on the Akash Decentralized Cloud. This model embodies the spirit of open-source and decentralized computing and leverages Creative Commons licensed data for its training. 

### Model Details

- **Model Name:** akash-thumper-v1 (AT1)
- **Model Type:** Text to Image
- **Training Platform:** [Akash Decentralized Cloud](https://akash.network)
- **Release License:** Open Rails License
- **Weights:** [huggingface](https://huggingface.co/thumperai/akash-thumper-v1/)



## Performance and Limitations

While AT1 represents a pioneering effort in the use of decentralized cloud resources for AI model training, it's important to note that its performance does not currently meet the performance of other SOTA text-to-image generation models such as SDXL, Pixart-Alpha, or Stable Diffusion.   This discrepancy in performance is largely attributed to the decision to use exclusively Creative Commons licensed data for training. Though this choice underscores our commitment to using open data sources and open-source principles, it has resulted in certain limitations in the model's output quality and diversity.

![example images](https://github.com/thumperai/at1/images/at1_image_examples.png)


## Usage


## Training on Akash 

- To train on akash you will need to deploy ray cluster using the [akash console] (https://console.akash.network) and the deployments under ./Deployments/gpuraycluster.yaml 
- training job scripts are listed under ./scripts and PixArt-alpa/script
- scripts should be launched with launch.py, launch.py is a wrapper around ray jobs python api to make it easier to upload src code to the ray cluster or rebuild the ray python environment without having to rebuild the docker container. 


## Example Notebook


## License

The `akash-thumper-v1` model is released under the Open Rails License, emphasizing its commitment to open-source principles and the promotion of Creative Commons Foundation models. Users are encouraged to contribute to the model's development, share their modifications, and help improve the performance and capabilities of AT1.

## Contributions and Support

For support and discussions, please visit our GitHub repository or join our community forums. Let's collaborate to push the boundaries of decentralized AI and open-source development.

---
