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
 - [x] preprocessing function
  - [x] split to sentences
  - [x] tokenize and write to file
  - [x] from file to binary
  - [x] consider having a process bar for preprocessing
  - [x] consider more verbose output
  - [x] reconcider change generate-test file as it could be written over
 - [x] Translation checkpointer
  - [x] consider having the possibility to pick up from where it left off
  - [x] needs to keep track to where are each split state
  - [x] will work with a DB with state and unique ids
 - [x] translation cpu
  -[x] write function that calls generate in python
 - [x] translation gpu
 - [x] function that would loop over all the splits
  - [x] Each time a process is finished, could launch the next one
  - [x] function without split index would run the next that was not started
 - [ x postprocessing function
  - [x] reconciliate sentences
  - [x] merge function if last split is translated
  - [x] delete everything that is not needed
- [ ] check sizes in tqdm GB vs len?
- [ ] Think of a way to work asynchronously (peprocessing while translating)
- [ ] limite verbose
- [ ] refactor code
- [ ] Test on real data
- [ ] run rye ci

- [x] improve demo size to 10K
 - [x] give example of gpu memory + batch info + time

