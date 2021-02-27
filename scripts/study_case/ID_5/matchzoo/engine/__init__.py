# `engine` dependencies span across the entire project, so it's better to
# leave this __init__.py empty, and use `from scripts.study_case.MatchZoo_py.matchzoo.engine.package import
# x` or `from scripts.study_case.MatchZoo_py.matchzoo.engine import package` instead of `from scripts.study_case.MatchZoo_py.matchzoo
# import engine`.
