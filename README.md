Place the photos in the top level ‘images’ folder.

- - In Terminal Window - -

1. cd /<location>/altTextProject
2. <YOUR_AI_AGENT_OF_CHOICE>_API_KEY=sk-ant-your-key-here .venv/bin/python generate_alt_gallery.py

————
For a permanent fix (so you never have to type it again), add it to your shell profile:
———-

echo 'export <YOUR_AI_AGENT_OF_CHOICE>_API_KEY=sk-ant-your-key-here' >> ~/.zshrc
source ~/.zshrc

After that you can just run:
.venv/bin/python generate_alt_gallery.py
.venv/bin/python generate_alt_gallery.py --artist "<artist name"
.venv/bin/python generate_alt_gallery.py --band "band name"

———-
Before running script on a new batch of images, delete your old images from both 'images' and 'site/images'
