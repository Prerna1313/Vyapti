FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter "sed '/Co-Authored-By: Claude/d'" -- --all
