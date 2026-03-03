"""
数据库操作模块
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

try:
    import pymysql
except ImportError:
    pymysql = None


class DatabaseManager:
    """MySQL 数据库管理器"""
    
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table: str = "facts",
        project_id: str = None
    ):
        """
        初始化数据库管理器
        
        Args:
            host: MySQL 主机
            port: MySQL 端口
            user: MySQL 用户名
            password: MySQL 密码
            database: 数据库名
            table: 表名
            project_id: 项目 ID（用于数据隔离）
        """
        if pymysql is None:
            raise RuntimeError("未安装 pymysql。请安装: pip install pymysql")
        
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.project_id = project_id
    
    def get_connection(self, with_database=True):
        """获取数据库连接"""
        if with_database:
            return pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                autocommit=True,
            )
        else:
            # 不指定数据库，用于创建数据库
            return pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                charset="utf8mb4",
                autocommit=True,
            )
    
    def create_database(self) -> None:
        """创建数据库（如果不存在）"""
        conn = self.get_connection(with_database=False)
        try:
            with conn.cursor() as cur:
                create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{self.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                cur.execute(create_db_sql)
                print(f"数据库 {self.database} 已就绪")
        finally:
            conn.close()
    
    def create_conversations_table(self) -> None:
        """创建 conversations 表"""
        self.create_database()
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS `conversations` (
                    `conversation_id` INT AUTO_INCREMENT PRIMARY KEY,
                    `project_id` VARCHAR(255) NOT NULL,
                    `text` TEXT NOT NULL,
                    `content_type` ENUM('conversation', 'file_chunk') DEFAULT 'conversation',
                    `source_identifier` VARCHAR(500),
                    `source_metadata` JSON,
                    `source_file` VARCHAR(255),
                    `indexed` BOOLEAN DEFAULT FALSE,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX `idx_project_id` (`project_id`),
                    INDEX `idx_content_type` (`content_type`),
                    INDEX `idx_source_identifier` (`source_identifier`),
                    INDEX `idx_indexed` (`indexed`),
                    INDEX `idx_created_at` (`created_at`)
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
                """
                cur.execute(create_table_sql)
                
                # 为现有表添加新字段（如果不存在）- 兼容 MySQL 5.7+
                # 检查并添加 content_type
                cur.execute("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'conversations' 
                    AND COLUMN_NAME = 'content_type'
                """, (self.database,))
                if cur.fetchone()[0] == 0:
                    cur.execute("""
                        ALTER TABLE `conversations` 
                        ADD COLUMN `content_type` 
                        ENUM('conversation', 'file_chunk') DEFAULT 'conversation' AFTER `text`
                    """)
                    print("  ✓ 已添加 content_type 字段")
                
                # 检查并添加 source_identifier
                cur.execute("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'conversations' 
                    AND COLUMN_NAME = 'source_identifier'
                """, (self.database,))
                if cur.fetchone()[0] == 0:
                    cur.execute("""
                        ALTER TABLE `conversations` 
                        ADD COLUMN `source_identifier` VARCHAR(500) AFTER `content_type`
                    """)
                    print("  ✓ 已添加 source_identifier 字段")
                
                # 检查并添加 source_metadata
                cur.execute("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'conversations' 
                    AND COLUMN_NAME = 'source_metadata'
                """, (self.database,))
                if cur.fetchone()[0] == 0:
                    cur.execute("""
                        ALTER TABLE `conversations` 
                        ADD COLUMN `source_metadata` JSON AFTER `source_identifier`
                    """)
                    print("  ✓ 已添加 source_metadata 字段")
                
                # 检查并添加索引 idx_content_type
                cur.execute("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'conversations' 
                    AND INDEX_NAME = 'idx_content_type'
                """, (self.database,))
                if cur.fetchone()[0] == 0:
                    cur.execute("CREATE INDEX `idx_content_type` ON `conversations` (`content_type`)")
                    print("  ✓ 已创建 idx_content_type 索引")
                
                # 检查并添加索引 idx_source_identifier
                cur.execute("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'conversations' 
                    AND INDEX_NAME = 'idx_source_identifier'
                """, (self.database,))
                if cur.fetchone()[0] == 0:
                    cur.execute("CREATE INDEX `idx_source_identifier` ON `conversations` (`source_identifier`)")
                    print("  ✓ 已创建 idx_source_identifier 索引")
                
                print(f"表 {self.database}.conversations 已就绪（含 metadata 字段）")
        finally:
            conn.close()
    
    def create_facts_table(self) -> None:
        """创建 facts 表（带 conversation_id 外键）"""
        # 先确保 conversations 表存在
        self.create_conversations_table()
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS `{self.table}` (
                    `fact_id` INT AUTO_INCREMENT PRIMARY KEY,
                    `project_id` VARCHAR(255) NOT NULL,
                    `content` TEXT NOT NULL,
                    `conversation_id` INT,
                    `image_url` VARCHAR(1024),
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX `idx_project_id` (`project_id`),
                    INDEX `idx_conversation_id` (`conversation_id`),
                    INDEX `idx_created_at` (`created_at`),
                    FOREIGN KEY (`conversation_id`) REFERENCES `conversations`(`conversation_id`) ON DELETE CASCADE
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
                """
                cur.execute(create_table_sql)
                print(f"表 {self.database}.{self.table} 已就绪")
        finally:
            conn.close()
    
    def insert_conversation(
        self,
        text: str,
        source_file: str = None,
        project_id: str = None,
        content_type: str = "conversation",
        source_identifier: str = None,
        source_metadata: dict = None
    ) -> int:
        """
        插入对话原文
        
        Args:
            text: 对话原文
            source_file: 源文件名（向后兼容）
            project_id: 项目 ID（如果未指定，使用初始化时的 project_id）
            content_type: 内容类型 'conversation' 或 'file_chunk'
            source_identifier: 来源标识符（对话：session_id，文件：file_path）
            source_metadata: 完整的 metadata JSON
        
        Returns:
            conversation_id
        """
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        # 如果提供了 metadata，自动提取 source_identifier
        if source_metadata and not source_identifier:
            if content_type == "conversation":
                source_identifier = source_metadata.get("session_id")
            elif content_type == "file_chunk":
                source_identifier = source_metadata.get("file_path")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                insert_sql = """
                INSERT INTO `conversations` 
                (project_id, text, content_type, source_identifier, source_metadata, source_file, indexed)
                VALUES (%s, %s, %s, %s, %s, %s, FALSE)
                """
                import json
                metadata_json = json.dumps(source_metadata, ensure_ascii=False) if source_metadata else None
                cur.execute(insert_sql, (pid, text, content_type, source_identifier, metadata_json, source_file))
                conversation_id = cur.lastrowid
                conn.commit()
                return conversation_id
        finally:
            conn.close()
    
    def update_conversation_indexed(self, conversation_id: int, indexed: bool = True) -> None:
        """
        更新对话的索引状态
        
        Args:
            conversation_id: 对话 ID
            indexed: 是否已索引
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                update_sql = "UPDATE `conversations` SET indexed = %s WHERE conversation_id = %s"
                cur.execute(update_sql, (indexed, conversation_id))
                conn.commit()
        finally:
            conn.close()
    
    def get_conversation_by_id(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 conversation_id 获取对话
        
        Args:
            conversation_id: 对话 ID
        
        Returns:
            对话信息字典
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT conversation_id, project_id, text, source_file, indexed, created_at FROM `conversations` WHERE conversation_id = %s",
                    (conversation_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                
                conversation_id, project_id, text, source_file, indexed, created_at = row
                return {
                    "conversation_id": int(conversation_id),
                    "project_id": str(project_id),
                    "text": str(text or ""),
                    "source_file": str(source_file or ""),
                    "indexed": bool(indexed),
                    "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
                }
        finally:
            conn.close()
    
    def insert_facts(
        self, 
        facts: List[Any], 
        project_id: str = None,
        conversation_id: int = None
    ) -> int:
        """
        插入 facts（关联到 conversation）
        
        Args:
            facts: facts 列表，可以是：
                - List[str]: 字符串列表（向后兼容）
                - List[dict]: 字典列表，每个元素包含 {'content': str, 'image_url': str(可选)}
            project_id: 项目 ID（如果未指定，使用初始化时的 project_id）
            conversation_id: 对话 ID（外键，通过此关联获取 source_file）
        
        Returns:
            插入的 facts 数量
        """
        if not facts:
            return 0
        
        # 使用传入的 project_id 或实例的 project_id
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                insert_sql = f"INSERT INTO `{self.table}` (project_id, content, conversation_id, image_url) VALUES (%s, %s, %s, %s)"
                count = 0
                for fact in facts:
                    # 支持两种格式
                    if isinstance(fact, dict):
                        content = fact.get('content', '').strip()
                        image_url = fact.get('image_url')
                    else:
                        content = fact.strip() if isinstance(fact, str) else str(fact).strip()
                        image_url = None
                    
                    if content:
                        cur.execute(insert_sql, (pid, content, conversation_id, image_url))
                        count += 1
                conn.commit()
                return count
        finally:
            conn.close()
    
    
    def get_recent_facts(self, project_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取最近的 facts（按 fact_id 降序）
        
        Args:
            project_id: 项目 ID
            limit: 返回数量限制
        
        Returns:
            facts 列表
        """
        pid = project_id or self.project_id
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                sql = f"""
                SELECT 
                    f.fact_id, 
                    f.project_id, 
                    f.content, 
                    f.conversation_id,
                    f.image_url,
                    c.text as conversation_text,
                    c.source_file,
                    f.created_at
                FROM `{self.table}` f
                LEFT JOIN `conversations` c ON f.conversation_id = c.conversation_id
                """
                
                if pid:
                    sql += " WHERE f.project_id = %s"
                
                sql += " ORDER BY f.fact_id DESC LIMIT %s"
                
                params = (pid, limit) if pid else (limit,)
                cur.execute(sql, params)
                
                rows = cur.fetchall()
                
                facts = []
                for fact_id, proj_id, content, conversation_id, image_url, conversation_text, source_file, created_at in rows:
                    facts.append({
                        "fact_id": int(fact_id),
                        "project_id": str(proj_id),
                        "content": str(content or ""),
                        "conversation_id": int(conversation_id) if conversation_id else None,
                        "image_url": image_url,
                        "conversation_text": str(conversation_text or ""),
                        "source_file": str(source_file or ""),
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
                    })
                
                # 反转顺序，使其按 fact_id 升序排列
                return list(reversed(facts))
        finally:
            conn.close()
    
    def get_facts_by_ids(self, fact_ids: List[int]) -> Dict[int, Dict[str, str]]:
        """
        根据 fact_ids 批量获取 facts（JOIN conversations 表，包含完整metadata）
        
        Args:
            fact_ids: fact_id 列表
        
        Returns:
            {fact_id: {'content': ..., 'conversation_text': ..., 'image_url': ..., 'project_id': ..., ...}} 字典
        """
        if not fact_ids:
            return {}
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                placeholders = ','.join(['%s'] * len(fact_ids))
                sql = f"""
                SELECT f.fact_id, f.content, f.conversation_id, f.image_url, 
                       c.text as conversation_text, c.project_id, c.content_type, c.created_at
                FROM `{self.table}` f
                LEFT JOIN `conversations` c ON f.conversation_id = c.conversation_id
                WHERE f.fact_id IN ({placeholders})
                """
                cur.execute(sql, list(fact_ids))
                rows = cur.fetchall()
                
                result = {}
                for fact_id, content, conversation_id, image_url, conversation_text, project_id, content_type, created_at in rows:
                    result[int(fact_id)] = {
                        'content': str(content or ""),
                        'conversation_id': conversation_id,
                        'image_url': image_url,
                        'conversation_text': str(conversation_text or ""),
                        'project_id': project_id,
                        'created_at': created_at.timestamp() if created_at else None,
                        'created_at_iso': created_at.isoformat() if created_at else None,
                        'content_type': content_type
                    }
                
                return result
        finally:
            conn.close()
    
    def get_all_facts(self, project_id: str = None) -> List[Dict[str, Any]]:
        """
        获取所有facts（用于构建BM25索引）
        
        Args:
            project_id: 项目ID（可选，如果不指定则使用初始化时的project_id）
        
        Returns:
            [{'fact_id': int, 'content': str, 'conversation_id': int}, ...]
        """
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                sql = """
                SELECT f.fact_id, f.content, f.conversation_id
                FROM `facts` f
                JOIN `conversations` c ON f.conversation_id = c.conversation_id
                WHERE c.project_id = %s AND c.indexed = 1
                """
                cur.execute(sql, (pid,))
                rows = cur.fetchall()
                
                return [
                    {
                        'fact_id': int(fact_id),
                        'content': str(content or ""),
                        'conversation_id': conversation_id
                    }
                    for fact_id, content, conversation_id in rows
                ]
        finally:
            conn.close()
    
    def get_unindexed_conversations_with_facts(self, project_id: str = None, metadata_filter: dict = None) -> List[Dict[str, Any]]:
        """
        获取未索引的 conversations 及其关联的 facts（短期记忆）
        
        Args:
            project_id: 项目 ID
            metadata_filter: 元数据过滤条件，例如 {'content_type': 'file_chunk'}
            
        Returns:
            列表，每个元素包含 conversation 和其对应的 facts
        """
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 1. 查询未索引的 conversations
                sql = """
                SELECT conversation_id, project_id, text, source_file, created_at, 
                       content_type, source_identifier, source_metadata
                FROM `conversations`
                WHERE project_id = %s AND indexed = FALSE
                """
                params = [pid]
                
                # 应用 metadata_filter
                if metadata_filter:
                    if 'content_type' in metadata_filter:
                        sql += " AND content_type = %s"
                        params.append(metadata_filter['content_type'])
                
                sql += " ORDER BY created_at DESC"
                
                cur.execute(sql, tuple(params))
                conversations = cur.fetchall()
                
                if not conversations:
                    return []
                
                # 2. 批量查询这些 conversations 的 facts
                conv_ids = [row[0] for row in conversations]
                placeholders = ','.join(['%s'] * len(conv_ids))
                facts_sql = f"""
                SELECT conversation_id, fact_id, content, image_url
                FROM `{self.table}`
                WHERE conversation_id IN ({placeholders})
                ORDER BY fact_id ASC
                """
                cur.execute(facts_sql, conv_ids)
                facts_rows = cur.fetchall()
                
                # 3. 组织数据：按 conversation_id 分组
                facts_by_conv = {}
                for conv_id, fact_id, content, image_url in facts_rows:
                    if conv_id not in facts_by_conv:
                        facts_by_conv[conv_id] = []
                    facts_by_conv[conv_id].append({
                        'fact_id': int(fact_id),
                        'content': str(content or ""),
                        'image_url': image_url
                    })
                
                # 4. 合并结果（并应用额外的 metadata 过滤）
                results = []
                for conv_id, proj_id, text, source_file, created_at, content_type, source_identifier, source_metadata_json in conversations:
                    # 解析 JSON metadata
                    import json
                    metadata = json.loads(source_metadata_json) if source_metadata_json else {}
                    
                    # 应用 metadata_filter（对 JSON 字段的过滤）
                    if metadata_filter:
                        # 检查是否所有过滤条件都满足（除了 content_type，已在 SQL 中过滤）
                        match = True
                        for key, value in metadata_filter.items():
                            if key == 'content_type':
                                continue  # 已在 SQL 中处理
                            if metadata.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue  # 跳过不匹配的记录
                    
                    results.append({
                        'conversation_id': int(conv_id),
                        'project_id': str(proj_id),
                        'text': str(text or ""),
                        'source_file': str(source_file or ""),
                        'created_at': created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                        'content_type': content_type,
                        'source_identifier': source_identifier,
                        'metadata': metadata,  # ← 添加 metadata
                        'facts': facts_by_conv.get(conv_id, [])
                    })
                
                return results
        finally:
            conn.close()
    
    def find_conversation_by_file_chunk(
        self, 
        project_id: str, 
        file_hash: str, 
        chunk_index: int
    ) -> Optional[int]:
        """
        通过 file_hash 和 chunk_index 查询 conversation_id（用于替换模式）
        
        Args:
            project_id: 项目 ID
            file_hash: 文件哈希值（metadata.file_hash）
            chunk_index: chunk 索引（metadata.chunk_index）
            
        Returns:
            conversation_id 或 None（未找到）
        """
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 通过 JSON 字段查询（返回最新的一条）
                sql = """
                SELECT conversation_id 
                FROM `conversations` 
                WHERE project_id = %s 
                  AND JSON_UNQUOTE(JSON_EXTRACT(source_metadata, '$.file_hash')) = %s
                  AND JSON_EXTRACT(source_metadata, '$.chunk_index') = %s
                ORDER BY conversation_id DESC
                LIMIT 1
                """
                cur.execute(sql, (pid, file_hash, chunk_index))
                result = cur.fetchone()
                
                if result:
                    return result[0]  # conversation_id
                return None
        finally:
            conn.close()
    
    def get_conversation_with_facts(
        self,
        conversation_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        查询单个 conversation 及其关联的 facts
        
        Args:
            conversation_id: 对话 ID
            
        Returns:
            字典包含 conversation 信息和 facts 列表，或 None（未找到）
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 1. 查询 conversation
                conv_sql = """
                SELECT conversation_id, project_id, text, source_file, created_at, 
                       content_type, source_identifier, source_metadata
                FROM `conversations`
                WHERE conversation_id = %s
                """
                cur.execute(conv_sql, (conversation_id,))
                conv_row = cur.fetchone()
                
                if not conv_row:
                    return None
                
                conv_id, proj_id, text, source_file, created_at, content_type, source_identifier, source_metadata_json = conv_row
                
                # 解析 metadata
                metadata = {}
                if source_metadata_json:
                    try:
                        metadata = json.loads(source_metadata_json)
                    except:
                        pass
                
                # 2. 查询关联的 facts
                facts_sql = f"""
                SELECT fact_id, content, image_url
                FROM `{self.table}`
                WHERE conversation_id = %s
                ORDER BY fact_id ASC
                """
                cur.execute(facts_sql, (conversation_id,))
                facts_rows = cur.fetchall()
                
                facts = []
                for fact_id, content, image_url in facts_rows:
                    facts.append({
                        'fact_id': fact_id,
                        'content': content,
                        'image_url': image_url
                    })
                
                return {
                    'conversation_id': conv_id,
                    'project_id': proj_id,
                    'text': text,
                    'source_file': source_file,
                    'created_at': created_at,
                    'content_type': content_type,
                    'source_identifier': source_identifier,
                    'metadata': metadata,
                    'facts': facts
                }
        finally:
            conn.close()
    
    def get_recent_conversations(
        self, 
        project_id: str = None, 
        limit: int = 5,
        metadata_filter: dict = None
    ) -> List[Dict[str, Any]]:
        """
        获取最近N轮对话（用于多轮对话上下文）
        
        Args:
            project_id: 项目 ID
            limit: 返回最近N轮对话
            metadata_filter: 元数据过滤条件
            
        Returns:
            列表，每个元素包含 conversation 信息（按 turn 倒序）
        """
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 查询最近N轮对话（按 turn 倒序）
                sql = """
                SELECT conversation_id, project_id, text, source_file, created_at, 
                       content_type, source_identifier, source_metadata
                FROM `conversations`
                WHERE project_id = %s
                """
                params = [pid]
                
                # 应用 metadata_filter
                if metadata_filter:
                    if 'content_type' in metadata_filter:
                        sql += " AND content_type = %s"
                        params.append(metadata_filter['content_type'])
                
                # 按 metadata 中的 turn 倒序，如果没有 turn 则按创建时间倒序
                sql += " ORDER BY "
                sql += "CAST(JSON_EXTRACT(source_metadata, '$.turn') AS UNSIGNED) DESC, "
                sql += "created_at DESC "
                sql += f"LIMIT {limit}"
                
                cur.execute(sql, tuple(params))
                conversations = cur.fetchall()
                
                if not conversations:
                    return []
                
                # 组织数据
                results = []
                for conv_id, proj_id, text, source_file, created_at, content_type, source_identifier, source_metadata_json in conversations:
                    # 解析 JSON metadata
                    import json
                    metadata = json.loads(source_metadata_json) if source_metadata_json else {}
                    
                    # 应用 metadata_filter（对 JSON 字段的过滤）
                    if metadata_filter:
                        # 检查是否所有过滤条件都满足
                        match = True
                        for key, value in metadata_filter.items():
                            if key == 'content_type':
                                continue  # 已在 SQL 中处理
                            if metadata.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    results.append({
                        'conversation_id': int(conv_id),
                        'project_id': str(proj_id),
                        'text': str(text or ""),
                        'source_file': str(source_file or ""),
                        'created_at': created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                        'content_type': content_type,
                        'source_identifier': source_identifier,
                        'metadata': metadata
                    })
                
                # 按 turn 升序返回（最早的对话在前）
                results.reverse()
                
                return results
        finally:
            conn.close()
    
    def get_indexed_conversations(self, project_id: str = None) -> List[int]:
        """
        获取已索引的对话ID列表
        
        Args:
            project_id: 项目 ID
            
        Returns:
            已索引的 conversation_id 列表
        """
        pid = project_id or self.project_id
        if not pid:
            raise ValueError("必须指定 project_id")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                sql = """
                SELECT conversation_id
                FROM `conversations`
                WHERE project_id = %s AND indexed = TRUE
                ORDER BY conversation_id ASC
                """
                cur.execute(sql, (pid,))
                rows = cur.fetchall()
                return [int(row[0]) for row in rows]
        finally:
            conn.close()
    
    def get_facts_count(self, project_id: str = None) -> int:
        """
        获取 facts 总数（可按项目过滤）
        
        Args:
            project_id: 项目 ID（如果未指定，使用初始化时的 project_id）
        
        Returns:
            facts 数量
        """
        pid = project_id or self.project_id
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                if pid:
                    cur.execute(f"SELECT COUNT(*) FROM `{self.table}` WHERE project_id = %s", (pid,))
                else:
                    cur.execute(f"SELECT COUNT(*) FROM `{self.table}`")
                return cur.fetchone()[0]
        finally:
            conn.close()
    
    def expand_facts_by_turn(
        self,
        seed_fact_ids: List[int],
        hop_distance: int = 1,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        基于对话轮次扩展 Facts（时序多跳）
        
        Args:
            seed_fact_ids: 种子 Fact IDs
            hop_distance: 跳数（1=±1轮，2=±2轮）
            direction: 扩展方向
                - "both": 双向（前后都扩展）
                - "forward": 向后（更晚的轮次）
                - "backward": 向前（更早的轮次）
        
        Returns:
            {
                'facts': List[Dict],  # 扩展的 Facts
                'conversations': List[Dict],  # 涉及的对话
                'turn_range': (min_turn, max_turn)
            }
        """
        if not seed_fact_ids:
            return {'facts': [], 'conversations': [], 'turn_range': (None, None)}
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                import json
                
                # Step 1: 获取种子 Facts 的 turns
                placeholders = ','.join(['%s'] * len(seed_fact_ids))
                cur.execute(f"""
                    SELECT DISTINCT c.source_metadata
                    FROM `{self.table}` f
                    JOIN `conversations` c ON f.conversation_id = c.conversation_id
                    WHERE f.fact_id IN ({placeholders})
                """, seed_fact_ids)
                
                seed_turns = []
                for (metadata_json,) in cur.fetchall():
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    turn = metadata.get('turn')
                    if turn is not None:
                        seed_turns.append(turn)
                
                if not seed_turns:
                    return {'facts': [], 'conversations': [], 'turn_range': (None, None)}
                
                min_seed_turn = min(seed_turns)
                max_seed_turn = max(seed_turns)
                
                # Step 2: 计算目标 turn 范围
                if direction == "both":
                    target_min_turn = min_seed_turn - hop_distance
                    target_max_turn = max_seed_turn + hop_distance
                elif direction == "forward":
                    target_min_turn = min_seed_turn
                    target_max_turn = max_seed_turn + hop_distance
                elif direction == "backward":
                    target_min_turn = min_seed_turn - hop_distance
                    target_max_turn = max_seed_turn
                else:
                    raise ValueError(f"Invalid direction: {direction}")
                
                # Step 3: 查询目标 turn 范围内的所有对话
                cur.execute("""
                    SELECT conversation_id, text, source_metadata, created_at
                    FROM `conversations`
                    WHERE project_id = %s
                      AND JSON_EXTRACT(source_metadata, '$.turn') IS NOT NULL
                      AND CAST(JSON_EXTRACT(source_metadata, '$.turn') AS UNSIGNED) >= %s
                      AND CAST(JSON_EXTRACT(source_metadata, '$.turn') AS UNSIGNED) <= %s
                    ORDER BY CAST(JSON_EXTRACT(source_metadata, '$.turn') AS UNSIGNED)
                """, (self.project_id, target_min_turn, target_max_turn))
                
                expanded_conversations = cur.fetchall()
                
                # Step 4: 获取这些对话的所有 Facts
                expanded_facts = []
                if expanded_conversations:
                    expanded_conv_ids = [row[0] for row in expanded_conversations]
                    conv_placeholders = ','.join(['%s'] * len(expanded_conv_ids))
                    
                    cur.execute(f"""
                        SELECT f.fact_id, f.content, f.conversation_id, f.image_url
                        FROM `{self.table}` f
                        WHERE f.conversation_id IN ({conv_placeholders})
                        ORDER BY f.fact_id
                    """, expanded_conv_ids)
                    
                    for fact_id, content, conv_id, image_url in cur.fetchall():
                        expanded_facts.append({
                            'fact_id': fact_id,
                            'content': content,
                            'conversation_id': conv_id,
                            'image_url': image_url
                        })
                
                # Step 5: 组织对话信息
                conversations = []
                for conv_id, text, metadata_json, created_at in expanded_conversations:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    conversations.append({
                        'conversation_id': conv_id,
                        'text': text,
                        'turn': metadata.get('turn'),
                        'created_at': created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
                    })
                
                return {
                    'facts': expanded_facts,
                    'conversations': conversations,
                    'turn_range': (target_min_turn, target_max_turn)
                }
        
        finally:
            conn.close()
    
    def expand_facts_by_time(
        self,
        seed_fact_ids: List[int],
        time_window_minutes: int = 5,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        基于时间窗口扩展 Facts（时序多跳）
        
        Args:
            seed_fact_ids: 种子 Fact IDs
            time_window_minutes: 时间窗口（分钟）
            direction: 扩展方向
                - "both": 双向（前后都扩展）
                - "forward": 向后（更晚的时间）
                - "backward": 向前（更早的时间）
        
        Returns:
            {
                'facts': List[Dict],  # 扩展的 Facts
                'conversations': List[Dict],  # 涉及的对话
                'time_range': (start_time, end_time)
            }
        """
        if not seed_fact_ids:
            return {'facts': [], 'conversations': [], 'time_range': (None, None)}
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                from datetime import timedelta
                
                # Step 1: 获取种子 Facts 的时间范围
                placeholders = ','.join(['%s'] * len(seed_fact_ids))
                cur.execute(f"""
                    SELECT MIN(c.created_at), MAX(c.created_at)
                    FROM `{self.table}` f
                    JOIN `conversations` c ON f.conversation_id = c.conversation_id
                    WHERE f.fact_id IN ({placeholders})
                """, seed_fact_ids)
                
                result = cur.fetchone()
                if not result or not result[0]:
                    return {'facts': [], 'conversations': [], 'time_range': (None, None)}
                
                min_time, max_time = result
                
                # Step 2: 计算目标时间范围
                time_delta = timedelta(minutes=time_window_minutes)
                
                if direction == "both":
                    target_start_time = min_time - time_delta
                    target_end_time = max_time + time_delta
                elif direction == "forward":
                    target_start_time = min_time
                    target_end_time = max_time + time_delta
                elif direction == "backward":
                    target_start_time = min_time - time_delta
                    target_end_time = max_time
                else:
                    raise ValueError(f"Invalid direction: {direction}")
                
                # Step 3: 查询目标时间范围内的所有对话
                cur.execute("""
                    SELECT conversation_id, text, source_metadata, created_at
                    FROM `conversations`
                    WHERE project_id = %s
                      AND created_at >= %s
                      AND created_at <= %s
                    ORDER BY created_at
                """, (self.project_id, target_start_time, target_end_time))
                
                expanded_conversations = cur.fetchall()
                
                # Step 4: 获取这些对话的所有 Facts
                expanded_facts = []
                if expanded_conversations:
                    expanded_conv_ids = [row[0] for row in expanded_conversations]
                    conv_placeholders = ','.join(['%s'] * len(expanded_conv_ids))
                    
                    cur.execute(f"""
                        SELECT f.fact_id, f.content, f.conversation_id, f.image_url, c.created_at
                        FROM `{self.table}` f
                        JOIN `conversations` c ON f.conversation_id = c.conversation_id
                        WHERE f.conversation_id IN ({conv_placeholders})
                        ORDER BY c.created_at, f.fact_id
                    """, expanded_conv_ids)
                    
                    for fact_id, content, conv_id, image_url, created_at in cur.fetchall():
                        expanded_facts.append({
                            'fact_id': fact_id,
                            'content': content,
                            'conversation_id': conv_id,
                            'image_url': image_url,
                            'created_at': created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
                        })
                
                # Step 5: 组织对话信息
                import json
                conversations = []
                for conv_id, text, metadata_json, created_at in expanded_conversations:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    conversations.append({
                        'conversation_id': conv_id,
                        'text': text,
                        'turn': metadata.get('turn'),
                        'created_at': created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
                    })
                
                return {
                    'facts': expanded_facts,
                    'conversations': conversations,
                    'time_range': (
                        target_start_time.isoformat() if hasattr(target_start_time, 'isoformat') else str(target_start_time),
                        target_end_time.isoformat() if hasattr(target_end_time, 'isoformat') else str(target_end_time)
                    )
                }
        
        finally:
            conn.close()
    
    def clear_facts(self, project_id: str = None) -> None:
        """
        清空 facts（可按项目清空）
        
        Args:
            project_id: 项目 ID（如果未指定，使用初始化时的 project_id；如果都没有，清空整个表）
        """
        pid = project_id or self.project_id
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                if pid:
                    cur.execute(f"DELETE FROM `{self.table}` WHERE project_id = %s", (pid,))
                    print(f"已清空项目 {pid} 的 facts")
                else:
                    cur.execute(f"TRUNCATE TABLE `{self.table}`")
                    print(f"表 {self.database}.{self.table} 已清空")
                conn.commit()
        finally:
            conn.close()

