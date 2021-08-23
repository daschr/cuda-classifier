#!/bin/bash

bench(){
	git checkout "$1" >/dev/null 2>&1 || return 
	make >/dev/null || return 

	local dur="$(./lscl rules trace /dev/null | sed -En 's/CLASSIFICATION took\W+(\w+)\W+.*/\1/p')"
   	
	echo "$dur"	
}

if (( $# == 0 )); then
		echo "Usage: $0 [name] [?pepend file]" >&2
		exit 1
fi

if (( $#==1 )); then
	echo "|#rules|#headers|type|$1|"
			echo "|------|--------|----|$(eval printf "%0.1s" -{1..${#1}})|"
else
	echo "$(head -n1 "$2")$1|"
	echo "$(head -n2 "$2"| tail -n1)$(eval printf "%0.1s" -{1..${#1}})|"
fi

line=3
i=0
branches=( "master" "async" "persistent" )
names=( "**simple**" "**async**" "**persistent**" )

for nheaders in 100000 1000000; do
		nrules=100
		for _ in {0..2}; do
				for i in {0..2}; do 
					./gen_cls --size $nrules --num_headers $nheaders --seed $(( RANDOM * RANDOM ))
					d=$(bench "${branches[$i]}")
					if (( $#==1 )); then
						printf "|%'d|%'d|%s|%'d μs|\n" "$nrules" "$nheaders" "${names[$i]}" "$d"
					else
						printf "%s%'d μs|\n" "$(head -n $line "$2" | tail -n1)" "$d"
					fi
					line=$((++line))
				done
				nrules=$((nrules*10))
		done
done
