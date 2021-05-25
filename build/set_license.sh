#!/bin/bash
for f in $(find ./wax -type f -name "*py");
do
  if [ "$(head -c11 $f)" = "# Copyright" ]
  then
    true
  else
    cat LICENSE_SHORT $f > $f.license && mv $f.license $f
  fi
done
