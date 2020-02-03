import json

def annotation_loader(filename):
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            r = {}
            r["info"] = data["info"]
            r["images"] = data["images"]
            r["licenses"] = data["licenses"]
            r["annotations"] = data["annotations"]
            r["categories"] = data["categories"]
            yield r


def get_img():
    filename = "../annotations/instances_val2014.json"
    data = annotation_loader(filename)
    for row in data:
        with open("images_val.json", 'w') as f:
            json.dump(row["images"], f)

def test():
    filename = "../annotations/instances_train2014.json"
    data = annotation_loader(filename)
    read=0
    for row in data:
        print(row["images"])
        read+=1
        if read == 3:
            break
    print(read)

if __name__ == '__main__':
    get_img()
