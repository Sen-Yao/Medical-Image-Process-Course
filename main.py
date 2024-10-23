import argparse
import torch
import yaml

from dataset import MedicalDataset
from models.basic import ThresholdSegmentationModel
from models.clustering import ColorSpace_Clustering, ColorSpace_Grad_Clustering, ColorSpace_Texture_Clustering, ResNet_Clustering
from models.graph import GraphCutSegmentation, RandomWalkSegmentation

from evaluate import evaluate_model

def main_work(args, cfg):
    dataset = MedicalDataset("Classic", 'dataset/Data', 256, 'Data')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    if args.model=="threshold":
        model = ThresholdSegmentationModel(cfg['threshold_segmentation']['threshold'])
    elif args.model=="cluster":
        if args.feature in ["RGB", "HSV", "LAB"]:
            model = ColorSpace_Clustering(cfg['cluster_segmentation']['k'], args.feature)
        elif args.feature in ["prewitt", "sobel", "canny"]:
            model = ColorSpace_Grad_Clustering(cfg['cluster_segmentation']['k'], "HSV", args.feature)
        elif args.feature in ["gabor", "laws"]:
            model = ColorSpace_Texture_Clustering(cfg['cluster_segmentation']['k'], "HSV", args.feature)
        elif args.feature in ["CNN"]:
            model = ResNet_Clustering(cfg['cluster_segmentation']['k'])
        else:
            print("Feature Error!")
            return 1
    elif args.model == "graph":
        model = GraphCutSegmentation(args.feature)
    else:
        print("Model Error!")
        return 1
    evaluate_model(dataset, model, int(args.batch_size))


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model", default="threshold", help="model"
    )

    parser.add_argument(
        "-f", "--feature", default="threshold", help="feature extraction"
    )

    parser.add_argument(
        "-c", "--config_path", default="config/default_config.yaml", help="the path of configuration"
    )

    parser.add_argument(
        "-b", "--batch_size", default="16", help="Batch size of image processing"
    )

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    
    main_work(args, cfg)