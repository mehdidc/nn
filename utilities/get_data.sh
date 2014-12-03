#!/bin/bash

mkdir -p ../data
scp -r cherti@lx2.lal.in2p3.fr:/exp/appstat/cherti/Projects/NN/data/*.* ../data
scp -r cherti@lx2.lal.in2p3.fr:/exp/appstat/cherti/Projects/NN/data/taxonomy ../data

echo "do yo want to download cifar data?"
select  result in Yes No
do
    if [ "$result" == "Yes" ]
    then
        scp -r cherti@lx2.lal.in2p3.fr:/exp/appstat/cherti/Projects/NN/data/cifar ../data
        break
    fi

    if [ "$result" == "No" ]
    then
        break
    fi
done

echo $result
