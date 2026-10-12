from pathlib import Path
from PIL import Image
import numpy as np
def fixture_sequence(root, count=3):
    seq = root / 'MOT20-01'
    (seq / 'img1').mkdir(parents=True)
    (seq / 'gt').mkdir()
    (seq / 'seqinfo.ini').write_text('[Sequence]\nname=MOT20-01\nimDir=img1\nframeRate=25\n'
                                   f'seqLength={count}\nimWidth=80\nimHeight=40\nimExt=.jpg\n')
    for number in range(1, count + 1):
        Image.fromarray(np.full((40, 80, 3), (number * 20) % 256, np.uint8)).save(seq / 'img1' / f'{number:06}.jpg')
    (seq / 'gt' / 'gt.txt').write_text('1,1,10,5,20,20,1,1,1\n1,2,50,5,20,20,1,7,1\n'
                                    '2,1,11,5,20,20,1,1,1\n3,1,12,5,20,20,1,1,1\n')
    return seq
