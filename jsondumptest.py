import json


def make_thing():
    transcript = {}
    transcript['Transcript'] = str("test of number one")
    transcript['words'] = []
    for i in range(len(transcript['Transcript'])):
        transcript['words'].append(transcript['Transcript'][i])
    return transcript


def write_file():
    dat = make_thing()
    fn = "./temp.json"
    print(fn)
    print(dat)
    with open(fn, 'w') as f:
        json.dump(dat, f, indent=4)
    f.close()


if __name__ == "__main__":
    print("doing it")
    write_file()
