"""
T13 — KnowledgeBaseManager 权限判断测试

重点：_database_info_accessible 的所有权限分支
"""

import pytest

from knowledge.kb_manager import KnowledgeBaseManager, KBNotFoundError

# ================================================================== 测试数据常量
# ==================================================================
SUPER_ADMIN = {"uid": "admin_001", "role": "superadmin", "department_id": None}
NORMAL_USER = {"uid": "user_123", "role": "user", "department_id": 5}
DEPT_STRING_USER = {"uid": "user_456", "role": "user", "department_id": "7"}  # 字符串类型部门
USER_NO_DEPT = {"uid": "user_789", "role": "user", "department_id": None}     # 没部门

class TestDatabaseInfoAccessible:
    """逐个覆盖 _database_info_accessible 的权限分支"""

    # ------------------------------------------------------------------ ① superadmin 全部放行
    class TestSuperadmin:
        def test_superadmin_always_returns_true(self):
            """superadmin 不看 share_config，任何库都有权"""
            db_info = {"created_by": "someone_else", "share_config": None}
            assert KnowledgeBaseManager._database_info_accessible(SUPER_ADMIN, db_info) is True

        def test_superadmin_with_empty_db_info(self):
            assert KnowledgeBaseManager._database_info_accessible(SUPER_ADMIN, {}) is True

    # ------------------------------------------------------------------ ② 自己创建的 = 有权
    class TestCreatorAccess:
        def test_creator_access_without_share_config(self):
            """创建者即使没 share_config 也有权"""
            db_info = {"created_by": "user_123"}
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is True

        def test_non_creator_without_share_config(self):
            """非创建者，share_config=None → 走默认 {"access_level": "global"} → 有权"""
            db_info = {"created_by": "someone_else"}
            # 默认 share_config 是 global 级别
            result = KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info)
            assert result is True  # 因为 DEFAULT_SHARE_CONFIG 是 global

    # ------------------------------------------------------------------ ③-a global 级
    class TestGlobalAccess:
        def test_global_access_all_users(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {"access_level": "global"},
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is True
            assert KnowledgeBaseManager._database_info_accessible(USER_NO_DEPT, db_info) is True

    # ------------------------------------------------------------------ ③-b department 级
    class TestDepartmentAccess:
        def test_same_department_accessible(self):
            """用户的 department_id 在 department_ids 列表里 → 有权"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": [5, 6, 7],
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is True  # dept=5 ✓

        def test_different_department_denied(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": [1, 2, 3],
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is False

        def test_user_without_department_id_denied(self):
            """用户没有 department_id → 直接返回 False"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": [5],
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(USER_NO_DEPT, db_info) is False

        def test_department_ids_accepts_string_numbers(self):
            """department_ids 里的数字字符串也能匹配（源码会转 int）"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": ["5", "6"],  # 字符串！
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is True

        def test_user_department_id_as_string_also_works(self):
            """user.department_id 是字符串 '7'，department_ids 是 [7] → 转 int 后匹配"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": [7],
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(DEPT_STRING_USER, db_info) is True

        def test_invalid_department_ids_returns_false(self):
            """department_ids 里有非数字字符串 → ValueError → 返回 False"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": ["not_a_number"],
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is False

        def test_empty_department_ids_denies(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "department",
                    "department_ids": [],
                    "user_uids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is False

    # ------------------------------------------------------------------ ③-c user 级
    class TestUserLevelAccess:
        def test_explicit_user_in_list_accessible(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "user",
                    "user_uids": ["user_123", "user_999"],
                    "department_ids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is True

        def test_user_not_in_list_denied(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "user",
                    "user_uids": ["other_user"],
                    "department_ids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is False

        def test_user_level_with_empty_list_denies(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "user",
                    "user_uids": [],
                    "department_ids": [],
                },
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is False

        def test_user_level_with_none_uid_denies(self):
            """user.uid 为空字符串 → user_uid 为空 → 返回 False"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "user",
                    "user_uids": ["user_123"],
                    "department_ids": [],
                },
            }
            user_no_uid = {"uid": "", "role": "user"}
            assert KnowledgeBaseManager._database_info_accessible(user_no_uid, db_info) is False

    # ------------------------------------------------------------------ 未知 access_level / 边界情况
    class TestEdgeCases:
        def test_unknown_access_level_returns_false(self):
            db_info = {
                "created_by": "someone_else",
                "share_config": {"access_level": "unknown_level_xyz"},
            }
            assert KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info) is False

        def test_share_config_empty_dict_falls_back_to_default(self):
            """share_config={} 是 falsy，源码用 or 回退到 DEFAULT_SHARE_CONFIG（global 级别）"""
            db_info = {"created_by": "someone_else", "share_config": {}}
            result = KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info)
            # {} or DEFAULT_SHARE_CONFIG → {"access_level": "global"} → True
            assert result is True

        def test_share_config_none_also_falls_back_to_default(self):
            """share_config=None 同样走 or 回退到 global"""
            db_info = {"created_by": "someone_else", "share_config": None}
            result = KnowledgeBaseManager._database_info_accessible(NORMAL_USER, db_info)
            assert result is True

        def test_user_uid_as_integer_is_converted_to_string(self):
            """user.uid 是整数 123 → 源码 str(user.get('uid')) → '123'"""
            db_info = {
                "created_by": "someone_else",
                "share_config": {
                    "access_level": "user",
                    "user_uids": ["123"],
                    "department_ids": [],
                },
            }
            user_int_uid = {"uid": 123, "role": "user"}
            assert KnowledgeBaseManager._database_info_accessible(user_int_uid, db_info) is True

# ================================================================== get_kb 基础方法（非权限）
class TestGetKb:
    def test_unsupported_kb_type_raises(self, tmp_path):
        mgr = KnowledgeBaseManager(work_dir=str(tmp_path))
        with pytest.raises(KBNotFoundError, match="不支持的知识库类型"):
            mgr._get_kb_for_database("nonexistent_type", "some_kb")
