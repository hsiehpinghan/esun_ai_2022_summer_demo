import sys
import json

from argparse import ArgumentParser

def main(args):
    
    print(json.loads(args.request))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--request',
                        type=str,
                        help='request')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    sys.argv = [sys.argv[0]]
    sys.argv += ['--request', '{"esun_uuid": "adefb7e8d9268b972b95b6fa53db93780b6b22fbf", "esun_timestamp": 1590493849, "sentence_list": ["喂 你好 密碼 我 要 進去", "喂 你好 密碼 哇 進去", "喂 你好 密碼 的 話 進去", "喂 您好 密碼 我 要 進去", "喂 你好 密碼 無法 進去", "喂 你好 密碼 waa 進去", "喂 你好 密碼 while 進去", "喂 你好 密碼 文化 進去", "喂 你好 密碼 挖 進去", "喂 您好 密碼 哇 進去"], "phoneme_sequence_list": ["w eI4 n i:3 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 t ax5 x w A:4 ts6 j ax n4 ts6_h y4", "w eI4 n j ax n2 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 u:2 f A:4 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 W AA1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 W AY1 L ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 w ax n2 x w A:4 ts6 j ax n4 ts6_h y4", "w eI4 n j ax n2 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 W IH1 L ts6 j ax n4 ts6_h y4"], "retry": 2}']
    main(args=parse_args())