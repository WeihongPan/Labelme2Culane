import os
import os.path as osp
from PIL import Image
import numpy as np
import cv2
import argparse
import yaml

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

def create_annotations(jsons, annotations):
    assert osp.exists(jsons)
    if not osp.exists(annotations):
        print('annotations file not exist, creating...')
        os.mkdir(annotations)

    for name in os.listdir(jsons):
        json_data = osp.join(jsons, name)
        save_path = osp.join(annotations, name[:-5])
        os.system("python json_to_dataset.py " + json_data + " -o " + save_path)

def generate_image_set(img_path, dataset, target_path, record_file, ids):
    target_path_full = osp.join(dataset, target_path)
    if not osp.exists(target_path_full):
        os.mkdir(target_path_full)

    lines = ""
    for id in ids:   
        # check.append(id)     
        name = osp.join(img_path, str(id).zfill(4)+'.jpg')
        img = cv2.imread(name)
        cv2.imwrite(osp.join(target_path_full, str(id).zfill(4)+'.png'), img)
        print(name)
        lines = lines + osp.join(target_path, str(id).zfill(4)+'.png')+'\n'

    with open(record_file, 'w') as f:
        f.write(lines)

def generate_label_set(annotations, dataset, img_path, label_path, record_file, ids, cls_num):
    label_path_full = osp.join(dataset, label_path)
    if not osp.exists(label_path_full):
        os.mkdir(label_path_full)
    lines = ""
    for id in ids:        
        anno = osp.join(annotations, str(id).zfill(4), 'label.png')
        label = Image.open(anno)
        label_data = np.asanyarray(label)
        pix = np.unique(label_data)
        label = Image.fromarray(label_data)
        label.save(osp.join(label_path_full, str(id).zfill(4)+'_label.png'))
        
        line = osp.join(img_path, str(id).zfill(4)+'.png') + ' ' + osp.join(label_path, str(id).zfill(4)+'_label.png')
        for i in range (1, cls_num+1):
            if i in pix:
                line = line + ' 1'
            else:
                line = line + ' 0'
        lines = lines + line + '\n'
    with open(record_file, 'w') as f:
        f.write(lines)

def generate_culane_annotations(annotations, dataset, img_path, ids, cls_num):
    for id in ids:
        label_path = osp.join(annotations, str(id).zfill(4), "label.png")
        label = Image.open(label_path)            
        w, h = label.size

        # 将行采样点按输入图像高度放缩
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f,culane_row_anchor)) # 根据提供的函数对指定序列做映射

        lines = "" 
        for lane_idx in range(1, cls_num+1):        
            line = ""
            for i,r in enumerate(sample_tmp):
                label_r = np.asarray(label)[int(round(r))] # 取出label图像中行坐标为int(round(r))的一行            
                pos = np.where(label_r == lane_idx)[0]                    
                if len(pos) == 0:            
                    continue
                pos = np.mean(pos)        
                line = line + str(pos) + " " + str(r) + " "                    
            if line != "":
                lines = lines + line + "\n"
        # print("({}): ".format(id))
        # print(lines)

        txt_path = osp.join(dataset, img_path, str(id).zfill(4)+'.lines.txt')
        with open(txt_path, 'w') as f:
            f.write(lines)

if __name__ == '__main__':
    config = yaml.load(open('config.yaml').read())

    img_path            = config['img_path']
    jsons               = config['jsons']
    annotations         = config['annotations']
    dataset             = config['dataset']
    culane_annotations  = config['culane_annotations']
    label_path              = config['label_path']
    train_img           = config['train_img']
    val_img             = config['val_img']
    test_img            = config['test_img']
    list_path           = config['list_path']
    train_gt            = config['train_gt']
    val_gt              = config['val_gt']
    train               = config['train']
    val                 = config['val']
    test                = config['test']
    test_rate           = config['test_rate']
    val_rate            = config['val_rate']
    cls_num             = config['cls_num']

    print('------------------confirm paths---------------------------')
    assert osp.exists(img_path)
    print('image path: %s' % img_path)
    assert osp.exists(jsons)
    print('json files: %s' % jsons)

    print('annotation files: %s' % annotations)

    assert osp.exists(dataset)
    print('dataset: %s' % dataset)

    culane_annotations = osp.join(dataset, culane_annotations)    
    print('culane annotation files: %s' % culane_annotations)
        
    print('label files: %s' % osp.join(dataset, label_path))
    print('train image files: %s' % osp.join(dataset, train_img))
    print('val image files: %s' % osp.join(dataset, val_img))
    print('test image files: %s' % osp.join(dataset, test_img))

    list_path = osp.join(dataset, list_path)    
    print('list files: %s' % list_path)
    
    train_gt = osp.join(list_path, train_gt)
    print('train ground truth path: %s' % train_gt)

    val_gt = osp.join(list_path, val_gt)
    print('val ground truth path: %s' % val_gt)

    train = osp.join(list_path, train)
    print('train set path: %s' % train)

    val = osp.join(list_path, val)
    print('val set path: %s' % val)

    test = osp.join(list_path, test)
    print('test set path: %s\n' % test)

    print('------------------create annotations----------------------')
    create_annotations(jsons, annotations)

    print('------------divide images into train/val/test-------------')
    dataset_size = len(os.listdir(jsons))
    print('dataset size: %d' % dataset_size)
    test_size = int(dataset_size * test_rate)
    val_size = int((dataset_size - test_size) * val_rate)
    train_size = dataset_size - test_size - val_size
    print('train size: %d' % train_size)
    print('val size: %d' % val_size)
    print('test size: %d' % test_size)
    
    random.seed()
    index = np.arange(dataset_size)
    np.random.shuffle(index)
    val_ids = index[:val_size]
    test_ids = index[val_size:val_size+test_size]
    train_ids = index[val_size+test_size:]

    print('------------------generate image set-----------------------')
    print('train set...')
    generate_image_set(img_path, dataset, train_img, train, train_ids)
    print('val set...')
    generate_image_set(img_path, dataset, val_img, val, val_ids)
    print('test set...')
    generate_image_set(img_path, dataset, test_img, test, test_ids)
    
    print('------------------generate label set-----------------------')
    print('train set...')
    generate_label_set(annotations, dataset, train_img, label_path, train_gt, train_ids, cls_num)
    print('val set...')
    generate_label_set(annotations, dataset, val_img, label_path, val_gt, val_ids, cls_num)

    print('--------------generate culane annotations------------------')
    print('train set...')
    generate_culane_annotations(annotations, dataset, train_img, train_ids, cls_num)
    print('val set...')
    generate_culane_annotations(annotations, dataset, val_img, val_ids, cls_num)
    print('test set...')
    generate_culane_annotations(annotations, dataset, test_img, test_ids, cls_num)