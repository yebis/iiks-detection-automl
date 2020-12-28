import sys

if __name__ == '__main__':

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    min_score_thresh = float(sys.argv[3])

    #endpoint='http://localhost:8501/v1/models/efficientdet:predict' #d4
    endpoint='http://localhost:8502/v1/models/efficientdet:predict' #d5

    import os
    import json
    import time
    import urllib.request
    import numpy as np
    from PIL import Image
    from inference import visualize_image_prediction

    img = Image.open(input_image_path)
    image_array = np.asarray(img)
    payload = {'instances': [{'image_arrays:0': image_array.tolist()}]}
    data = json.dumps(payload).encode("utf-8")

    time_start = time.time()

    request = urllib.request.Request(url=endpoint, data=data, method='POST')
    response = urllib.request.urlopen(request)
    content = response.read()
    response.close()

    time_end = time.time()    

    response = json.loads(content.decode())
    prediction = response['predictions'][0]
    
    for box in prediction:
        _score = box[5]
        _class = box[6]
        if _class == 1.0 and _score >= min_score_thresh:
            print('person: {}%'.format(int(_score * 100)))

    prediction = np.asarray(prediction)

    img = visualize_image_prediction(
        image_array,
        prediction,
        min_score_thresh=min_score_thresh)
    Image.fromarray(img).save(output_image_path)

    print('')
    print(endpoint)
    print('min_score_thresh:{}'.format(min_score_thresh))
    print(':elasped {:.1f} seconds'.format(time_end - time_start))


