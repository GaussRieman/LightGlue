import base64
from hashlib import md5
import sys

import pandas as pd
sys.path.append("/datadrive/codes/opensource/features/LightGlue")

from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd, load_numpy_image
import numpy as np
import cv2
import json
import time
import retrying
import requests
import multiprocessing as mp
import tqdm



def upload_image(image: np.ndarray):

    # send a request using PUT method with a given header
    filesvc_add_api = "https://fileman-dev.azurewebsites.net/api/add/base64"
    filesvc_add_api = "https://fileman.clobotics.cn/api/add/base64"
    filesvs_file_api = "https://fileman.clobotics.cn/api/file/"

    headers = {
        'FileManAPIAccessToken': 'Q2xvYm90aWNzLlJldGFpbC5CaXpNYW4uRmxvd01hbg==',
        'Content-Type': 'application/json'
    }
    
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    base64_str = base64.b64encode(im_buf_arr.tobytes()).decode('utf-8')
    
    img_name = md5(image.tobytes()).hexdigest()
    data = {
        "AccessToken":"Q2xvYm90aWNzLlJldGFpbC5CaXpNYW4uRmxvd01hbg==",
        "FileContent": base64_str,
        "UploadFileInfo":{
            "Name":'test.jpg'
        }
    }

    resp = requests.put(filesvc_add_api, headers=headers, json=data)
    decode_str = resp.content.decode('utf-8')
    res_json = json.loads(decode_str)
    file_id = res_json['FileId']
    file_url = filesvs_file_api + file_id

    return file_url



@retrying.retry(wait_fixed=2000, stop_max_attempt_number=3)
def download_image(url: str):
    try:
        content = requests.get(url).content
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error: ", e)
        raise e
    
    

def draw_pts(img: np.ndarray, pts: np.ndarray):
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (0, 255, 0), 3)
    return img



def process_image(src_img: np.ndarray, rectified_img: np.ndarray, rectified_pts: np.ndarray,
                  extractor, matcher):
    try:
        img0 = load_numpy_image(src_img).cuda()
        img1 = load_numpy_image(rectified_img).cuda()
        
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        print("feats0: ", feats0['keypoints'].shape)
        print("feats1: ", feats1['keypoints'].shape)
        
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]
        
        src_pts = np.float32(points0.cpu().numpy()).reshape(-1, 1, 2)
        dst_pts = np.float32(points1.cpu().numpy()).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # reverse the transformation
        img = cv2.warpPerspective(rectified_img, np.linalg.inv(M), (src_img.shape[1], src_img.shape[0]))
        reverse_pts = cv2.perspectiveTransform(np.float32([rectified_pts]), np.linalg.inv(M))
        return img, reverse_pts
    except Exception as e:
        print("Error: ", e)
        h,w = rectified_img[0:2]
        
        return rectified_img, rectified_pts



def process_df(df: pd.DataFrame, shard:str, extractor, matcher):
    t0 = time.time()
    img_urls = df['image'].unique()
    print(f"Total images: {len(img_urls)}")
    
    df_res = pd.DataFrame(columns=['img_url', 'sku_code', 'x', 'y'])
    
    for img_url in tqdm.tqdm(img_urls):
        try:
            base_name = img_url.split("/")[-1]
            base_name = base_name.split(".")[0]
            img_df = df[df['image'] == img_url]
            
            sku_codes = img_df['sku_code'].values
            src_img_url = img_url
            rectified_img_url = img_df['stitched_image'].values[0]
            
            src_img = download_image(src_img_url)
            print("src_img: ", src_img.shape)
            rectified_img = download_image(rectified_img_url)
            print("rectified_img: ", rectified_img.shape)
            xs = img_df['x'].values
            ys = img_df['y'].values
            xs = xs*rectified_img.shape[1]
            ys = ys*rectified_img.shape[0]
            rectified_pts = np.array(list(zip(xs, ys)))
            
            img, reverse_pts = process_image(src_img, rectified_img, rectified_pts, extractor, matcher)
            new_img_url = upload_image(img)
            
            reverse_pts = reverse_pts.squeeze()
            reverse_pts = reverse_pts.astype(np.int32)
            
            for i in range(len(sku_codes)):
                df_res.loc[len(df_res)] = [new_img_url, sku_codes[i], reverse_pts[i][0], reverse_pts[i][1]]
        except Exception as e:
            print("Error: ", e)
            continue


    df_res.to_csv(f"output2/data_shard_{shard}.csv", index=False)
    t1 = time.time()
    print("Processing time: ", (t1 - t0))

        # img = draw_pts(img, reverse_pts)
        # cv2.imwrite(f"output/{base_name}.jpg", img)



def process_csv(csv_path: str, extractor, matcher):
    df = pd.read_csv(csv_path)
    print("Total boxes: ", len(df), df.columns)
    
    # process by shards
    SHARD_SIZE = 10000
    shards = len(df)//SHARD_SIZE + 1
    print("Total shards: ", shards)

    for i in range(shards):
        print(f"Processing shard: {i}")
        start = i*SHARD_SIZE
        end = min((i+1)*SHARD_SIZE, len(df))
        df_shard = df.iloc[start:end]
        process_df(df_shard, str(i), extractor, matcher)
    

def main():
    # ALIKED+LightGlue
    extractor_aliked = ALIKED(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher_aliked = LightGlue(features='aliked').eval().cuda()  # load the matcher
    csv_path = "/datadrive/codes/opensource/features/LightGlue/data/reverse/extract/sub_1.csv"
    # csv_path = "/datadrive/codes/opensource/features/LightGlue/data/reverse/LM_sku_locations.csv"
    process_csv(csv_path, extractor_aliked, matcher_aliked)
    
    

if __name__ == "__main__":
    main()