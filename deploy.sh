#!/bin/bash

# Borrowed from vana-sdxl-training-cog

# Check if vana-ai-utils-py directory already exists
if [ -d "vana-ai-utils-py" ]; then
  rm -rf vanautils
fi

# Check if login flag is passed
if [[ "$*" == *--login* ]]; then
  cog login
  echo "Cog is now logged in"
fi

if [[ "$*" == *--prune* ]]; then
  echo "Pruning docker images"
  docker image prune
fi

# if [[ $(git status --porcelain) ]]; then
#   echo "There are unsaved changes in the Git repository. Please commit or stash your changes before continuing."
#   rm -rf vanautils
#   exit 1
# fi

cp -r ../vana-ai-utils-py/vanautils .

# Push to dev or prod
if [[ "$*" == *--dev* ]]; then
  if [[ "$*" == *--rebuild-base* ]]; then
    python ./scripts/setup.py
  fi
  # TODO: What's the point of dev and prod if they're both pushing to the same thing?
  cog push r8.im/vana-com/vana-llm-experiment-1

  # TODO: Figure out whether we need to tag + push to git. Not sure why we did this in the first place.
  # if [ $? -eq 0 ]; then

  #   echo "Successfully pushed image to Replicate."
  #   DEV_TRAINING_VERSION=$(curl -s -H "Authorization: Token adc90bfbb26659edb04da65004b6588ff2fc64f4" https://api.replicate.com/v1/models/vana-com/vana-lora-training-dev | grep -o '"id":"[^"]*' | cut -d '"' -f 4)
  #   git tag vana-com/vana-lora-training-dev
  #   git tag $DEV_TRAINING_VERSION
  #   git push origin vana-com/vana-lora-training-dev
  #   git push origin $DEV_TRAINING_VERSION
  # fi
fi

if [[ "$*" == *--prod* ]]; then
  # Generate 4 random numbers
  random_numbers=$(od -vAn -N4 -tu1 < /dev/urandom | tr -d ' ' | cut -c 1-4)

  # Prompt user to repeat back the numbers
  read -p "To deploy to prod, please repeat back the following 4 numbers in order:  $(echo -n $random_numbers) " user_numbers

  # Check if user entered the correct numbers
  if [ "$user_numbers" == "$random_numbers" ]; then
    python ./scripts/setup.py
    cog push r8.im/vana-com/vana-llm-experiment-1

  else
    echo "Incorrect numbers. Deployment to prod aborted."
    rm -rf vanautils
    exit 1
  fi
fi

# Remove vana-ai-utils-py directory
rm -rf vanautils