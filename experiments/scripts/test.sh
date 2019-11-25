#!/bin/bash

set -x # 调试，设置-x选项后，之后执行的每一条命令，都会显示的打印出来
set -e # 脚本执行的时候出现返回值为非零，整个脚本 就会立即退出 

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

OIFS=$IFS
IFS='a'  # 分隔符 a，STEP输入为[32a72...]，STEPSIZE=[320000,720000,...]
STEP="$4"
STEPSIZE="["
for i in $STEP; do
  STEPSIZE=${STEPSIZE}"${i}0000,"  # 字符串连接，STEPSIEZE + i + "0000"
done
STEPSIZE=${STEPSIZE}"]" 
IFS=$OIFS

ITERS=${5}0000

array=( $@ )  # 将全部输入参数作为一个数组array
len=${#array[@]} # '#'为计算长度  array[@] 取出array中的全部元素   计算array的长度
EXTRA_ARGS=${array[@]:5:$len} # 取出下标为5到len - 1的元素 
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_} # 在EXTRA_ARGS的元素之间添加"_"

case ${DATASET} in
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    declare -a TEST_IMDBS=("coco_2014_minival")  # 关联数组 key-value 字符串为下标
    ;;
  vg)
    TRAIN_IMDB="visual_genome_train_5"
    declare -a TEST_IMDBS=("visual_genome_test_5" "visual_genome_val_5")
    ;;
  ade)
    TRAIN_IMDB="ade_train_5"
    declare -a TEST_IMDBS=("ade_mtest_5" "ade_mval_5")
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
 
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then  #[[]] 判断条件是否成立 -z str 判断str是否为空
    EXTRA_ARGS_SLUG=${EXTRA_ARGS_SLUG}_${4}_${5}
else
    EXTRA_ARGS_SLUG=${4}_${5}
fi

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG") # &程序后台运行 > 输入定向符号   tee 内容重定向 -a追加到 LOG中
echo Logging output to "$LOG"

set +x # 关闭调试模式
NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_iter_${ITERS}.ckpt
set -x

for TEST_IMDB in "${TEST_IMDBS[@]}"
do
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --visualize \
    --set ${EXTRA_ARGS}
done

# add
./experiments/scripts/test.sh $@
