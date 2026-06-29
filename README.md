Place the photos in the top level ‘images’ folder.

- - In Terminal Window - -

1. cd /<location>/altTextProject
2, ANTHROPIC_API_KEY=sk-ant-your-key-here .venv/bin/python generate_alt_gallery.py

————
For a permanent fix (so you never have to type it again), add it to your shell profile:
———-

echo 'export ANTHROPIC_API_KEY=sk-ant-your-key-here' >> ~/.zshrc
source ~/.zshrc

After that you can just run:
.venv/bin/python generate_alt_gallery.py

———-
Be sure to delete existing photos from both images folders before running script again.
