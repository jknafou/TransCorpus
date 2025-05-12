# TODO
- [ ] create a pypi package
  - [ ] create pypi login and token

- [ ] preview:
 - [ ] add feature that allows split preview
- [ ] preview-test:
 - [ ] write test function, with a few thinking in :
  - [ ] translation side by side with text of different sizes?
  - [ ] what if no ids are given?
  - [ ] shouldn't we give two types of ids, let's say I want to display a pmid, I don't want to look for it...ids, let's say I want to display a pmid, I don't want to look for it...

- [ ] translate:
 - [x] create a function that would split huge corpus consistantly (fast)
 - [ ] preprocessing function
  - [x] split to sentences
  - [x] tokenize and write to file
  - [x] from file to binary
  - [ ] consider having a process bar
  - [ ] consider more verbose output
  - [ ] consider having the possibility to pick up from where it left off
  - [ ] reconcider change generate-test file as it could be written over
 - [x] translation cpu
  -[x] write function that calls generate in python
 - [x] translation gpu
 - [ ] postprocessing function
  - [x] reconciliate sentences
  - [ ] merge function if last split is translated
  - [ ] delete everything that is not needed
- [ ] run rye ci

- [ ] improve demo size to 10K
 - [ ] give example of gpu memory + batch info + time
- [ ] Test on real data

