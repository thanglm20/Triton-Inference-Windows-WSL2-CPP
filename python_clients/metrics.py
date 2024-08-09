

import requests
import argparse



def main(opt):
    print("Requesting to url: ", opt.url)
    response = requests.get(opt.url)
    print(response.content)
    # Print json data using loop
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument('--url', type=str, default='http://localhost:8002/metrics', help='path url metrics')
    opt = parser.parse_args()
    print(opt)
    main(opt)