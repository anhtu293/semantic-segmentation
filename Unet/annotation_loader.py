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


def test():
    filename = "../annotations/instances_train2014.json"
    data = annotation_loader(filename)
    read=0
    for row in data:
        print(row["annotations"])

        read+=1
        if read == 3:
            break
    print(read)

if __name__ == '__main__':
    test()
