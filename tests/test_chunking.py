"""chunking 模块回归测试。

重点：deep_merge 曾经在 for 循环内提前 return（只合并 override 的第一个 key），
这里用多 key 场景固定住正确行为。
"""

from knowledge.chunking import deep_merge


def test_deep_merge_merges_all_keys():
    """override 里的多个 key 都应被合并，而不是只合并第一个。"""
    base = {"a": 1, "b": {"x": 1}, "c": 3}
    override = {"b": {"y": 2}, "c": 30, "d": 4}

    result = deep_merge(base, override)

    assert result["a"] == 1              # 未覆盖的保留
    assert result["b"] == {"x": 1, "y": 2}  # 嵌套字典合并
    assert result["c"] == 30             # 覆盖
    assert result["d"] == 4              # 新增


def test_deep_merge_nested_recursive():
    """多层嵌套也能递归合并。"""
    base = {"a": {"b": {"c": 1}}}
    override = {"a": {"b": {"d": 2}}}

    result = deep_merge(base, override)

    assert result["a"]["b"] == {"c": 1, "d": 2}


def test_deep_merge_does_not_mutate_base():
    """deep_merge 是对 base 的深拷贝，不修改入参。"""
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}

    result = deep_merge(base, override)

    assert result["a"] == {"x": 1, "y": 2}
    assert base["a"] == {"x": 1}
