import json

masks_dicts = []
images_dicts = []

categories_dicts = [];
category_dict = {}
category_dict['id'] = 1
category_dict['name'] = 'nuclei'
categories_dicts.append(category_dict)

for image_id in dataset_train.image_ids:
    train_id = dataset_train.image_info[image_id]['path'].split('/')[-1][:-4] # id
    masks = dataset_train.load_mask(image_id)[0]
    
    image_dict['id'] = image_id
    image_dict['width'] = masks.shape[1]
    image_dict['height'] = masks.shape[0]
    image_dict['file_name'] = train_id+'.png'
    
    images_dicts.append(image_dict)
    
    bboxes = extract_bboxes(masks)
    
    mask_dict = {}
    for i in range(masks.shape[2]):
        mask_dict['id'] = image_id*100000+i
        mask_dict['image_id'] = image_id
        mask_dict['category_id'] = 1
        mask_dict['segmentation'] = prob_to_rles(masks[:,:,i])
        mask_dict['area'] = np.count_nonzero(masks[:,:,i])
        mask_dict['bbox'] = bboxes[i]
        
        mask_dict['image_longid'] = train_id
        masks_dicts.append(mask_dict)

all_dict = {}
all_dict['type'] = 'instances'
all_dict['images'] = images_dicts
all_dict['annotations'] = masks_dicts
all_dict['categories'] = categories_dicts

with open('nucleidata.json', 'w') as outfile:
    json.dump(all_dict, outfile)
