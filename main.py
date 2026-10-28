"""AstrBot 插件：给角色一套固定的社会关系（朋友 / 家人 / 同事）。

解决的问题：AI 的世界里只有"你"和"她"两个人，她不会提到任何别人，
所以她的生活是扁的、没有外部世界。

做法：
1. 按人格设定生成 3-6 个固定 NPC（名字、关系、性格、当前近况）；
2. 每轮对话把这份名单注入上下文，明确要求「只能用这些人」；
3. 每天让其中 1-2 个人的近况往前推一小步，让世界是活的而不是快照。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

FALLBACK_FRIENDS = [
    {"name": "小周", "relation": "大学室友", "traits": "说话直，爱吐槽", "recent": "面试连挂三次，正在考虑转行"},
    {"name": "妈妈", "relation": "母亲", "traits": "操心，嘴硬心软", "recent": "最近迷上广场舞，天天催我早睡"},
    {"name": "阿凯", "relation": "同事", "traits": "爱吹牛，人其实不坏", "recent": "刚买了个新相机，天天在群里发照片"},
]


def _extract_json(text: str) -> Any:
    if not text:
        return None
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:  # noqa: BLE001
                continue
    return None


class SocialCirclePlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}

        def cfg(key: str, default):
            value = self.config.get(key, default)
            return default if value is None else value

        self.enable_inject: bool = bool(cfg("enable_inject", True))
        self.friend_count: int = int(cfg("friend_count", 4) or 4)
        self.evolve_daily: bool = bool(cfg("evolve_daily", True))
        self.circle_hint: str = str(cfg("circle_hint", "") or "").strip()
        self.inject_max_chars: int = int(cfg("inject_max_chars", 600) or 600)
        self.auto_regenerate: bool = bool(cfg("auto_regenerate", True))

        try:
            data_dir = Path(StarTools.get_data_dir())
        except Exception:  # noqa: BLE001
            data_dir = Path("data/plugin_data/astrbot_plugin_social_circle")
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.state_path = data_dir / "circle.json"

        self.state: dict[str, Any] = self._load()
        self._persona_cache: dict[str, str] = {}
        self._generating = False
        self._evolving = False

        count = len(self.state.get("friends") or [])
        logger.info(
            f"[Social] 已加载：好友数={count or '未生成'} "
            f"注入={'开' if self.enable_inject else '关'} "
            f"每日演化={'开' if self.evolve_daily else '关'}",
        )

    # ---------------- 存储 ----------------

    def _load(self) -> dict[str, Any]:
        default = {"friends": [], "generated_at": "", "persona_hash": "", "last_evolve": ""}
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in default.items():
                        data.setdefault(k, v)
                    return data
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Social] 读取失败，使用空档案: {exc}")
        return default

    def _save(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps(self.state, ensure_ascii=False, indent=2),
                encoding="utf-8",
                newline="\n",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Social] 保存失败: {exc}")

    # ---------------- 工具 ----------------

    def _umo(self, event: AstrMessageEvent) -> str:
        try:
            umo = getattr(event, "unified_msg_origin", "")
            if umo:
                return str(umo)
        except Exception:  # noqa: BLE001
            pass
        try:
            return str(event.get_session_id())
        except Exception:  # noqa: BLE001
            return "unknown"

    def _persona_text(self, umo: str) -> str:
        if self._persona_cache.get(umo):
            return self._persona_cache[umo]
        try:
            conf = self.context.get_config()
        except Exception:  # noqa: BLE001
            return ""
        if not conf:
            return ""
        for key in ("persona", "personas"):
            try:
                items = conf.get(key) or []
            except Exception:  # noqa: BLE001
                continue
            if isinstance(items, str):
                return items.strip()
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    for f in ("system_prompt", "prompt", "content", "text"):
                        val = item.get(f)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                elif isinstance(item, str) and item.strip():
                    return item.strip()
        return ""

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]

    async def _provider(self):
        try:
            return self.context.get_using_provider()
        except Exception:  # noqa: BLE001
            return None

    # ---------------- 生成 ----------------

    async def _generate(self, persona: str) -> list[dict[str, Any]]:
        """按人设生成一套社会关系。失败时返回内置兜底名单。"""
        provider = await self._provider()
        if not provider:
            logger.warning("[Social] 没有可用 LLM，使用内置兜底名单")
            return [dict(f) for f in FALLBACK_FRIENDS]

        n = max(1, min(8, self.friend_count))
        prompt = (
            f"根据下面这个角色的人设，为她设计 {n} 个社会关系（朋友、家人、同事等）。\n\n"
            f"人设：\n{persona[:1500] if persona else '（未提供，按一个二十多岁的普通年轻人设计）'}\n\n"
            "要求：\n"
            "1. 名字用中文常见名，不要用「小明」「小红」这类模板名；\n"
            "2. 关系要多样，至少包含 1 个家人、1 个好友、1 个弱关系（同事/同学/网友）；\n"
            "3. 每个人的近况必须具体、有细节、能引发话题——"
            "不要写「在上班」，要写「刚被调去做一个没人想接的项目」；\n"
            "4. 不要和角色人设冲突，也不要全都是同龄同性；\n"
            "5. traits 是 2-3 个性格关键词，recent 是一句话（30 字以内）。\n\n"
            '只输出 JSON 数组，不要解释：[{"name":"小周","relation":"大学室友",'
            '"traits":"说话直，爱吐槽","recent":"面试连挂三次，正在考虑转行"}]'
        )
        if self.circle_hint:
            prompt += f"\n额外要求：{self.circle_hint}"

        try:
            resp = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                system_prompt="你擅长为虚构角色设计真实可信的社会关系网络。只输出 JSON。",
            )
            data = _extract_json((getattr(resp, "completion_text", "") or "").strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Social] 生成社会链失败: {exc}")
            data = None

        friends: list[dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                friends.append(
                    {
                        "name": name[:12],
                        "relation": str(item.get("relation") or "").strip()[:16],
                        "traits": str(item.get("traits") or "").strip()[:40],
                        "recent": str(item.get("recent") or "").strip()[:60],
                        "updated": datetime.now().strftime("%Y-%m-%d"),
                    }
                )

        if not friends:
            friends = [dict(f) for f in FALLBACK_FRIENDS]
        return friends

    async def _ensure_circle(self, umo: str, force: bool = False) -> list[dict[str, Any]]:
        """确保有一份可用名单；人设变了会自动重生。"""
        persona = self._persona_text(umo)
        phash = self._hash(persona)
        friends = self.state.get("friends") or []

        if friends and not force:
            old_hash = self.state.get("persona_hash") or ""
            if not old_hash or old_hash == phash or not self.auto_regenerate:
                return friends
            logger.info("[Social] 检测到人设变化，重新生成社会链")

        if self._generating:
            return friends or [dict(f) for f in FALLBACK_FRIENDS]

        self._generating = True
        try:
            friends = await self._generate(persona)
            self.state["friends"] = friends
            self.state["persona_hash"] = phash
            self.state["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save()
            names = "、".join(f["name"] for f in friends)
            logger.info(f"[Social] 社会链已生成：{names}")
        finally:
            self._generating = False
        return friends

    async def _evolve(self) -> None:
        """让 1-2 个人的近况往前推一小步。一天只做一次。"""
        friends = self.state.get("friends") or []
        if not friends:
            return

        provider = await self._provider()
        if not provider:
            return

        prompt = (
            "下面是某个角色的社会关系，以及他们当前的近况。"
            "假设时间过去了几天，请让其中 1-2 个人的近况往前推进一步。\n\n"
            f"{json.dumps(friends, ensure_ascii=False)}\n\n"
            "要求：\n"
            "1. 只能推进已有的故事线，不要推翻重来，也不要凭空多出新人物；\n"
            "2. 变化要小而具体，像真实生活：面试有结果了、跟对象吵架了、养的猫生病了；\n"
            "3. 大部分人保持原样不动；\n"
            "4. 名字、关系、性格一律不许改，只改 recent 字段；\n"
            "5. 推进要合理——不要每次都往坏的方向走。\n\n"
            "只输出完整的 JSON 数组（包含所有人，未变的保持原文）：\n"
            '[{"name":"小周","relation":"大学室友","traits":"说话直，爱吐槽","recent":"..."}]'
        )

        try:
            resp = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                system_prompt="你擅长写出有连续性的日常小事件。只输出 JSON。",
            )
            data = _extract_json((getattr(resp, "completion_text", "") or "").strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Social] 近况演化失败: {exc}")
            return

        if not isinstance(data, list) or len(data) != len(friends):
            logger.info("[Social] 演化结果结构不对，本轮跳过")
            return

        updated = []
        today = datetime.now().strftime("%Y-%m-%d")
        for old, new in zip(friends, data):
            if not isinstance(new, dict) or new.get("name") != old.get("name"):
                updated = []
                break
            recent = str(new.get("recent") or "").strip()[:60] or old.get("recent", "")
            updated.append(
                {
                    "name": old["name"],
                    "relation": old.get("relation", ""),
                    "traits": old.get("traits", ""),
                    "recent": recent,
                    "updated": today if recent != old.get("recent") else old.get("updated", ""),
                }
            )

        if updated:
            self.state["friends"] = updated
            self.state["last_evolve"] = today
            self._save()
            logger.info("[Social] 社会链近况已推进")

    # ---------------- 渲染 ----------------

    def _render(self, friends: list[dict[str, Any]]) -> str:
        lines = ["【你的社会关系（只能用这些人）】"]
        for f in friends:
            line = f"· {f['name']}"
            if f.get("relation"):
                line += f"｜{f['relation']}"
            if f.get("traits"):
                line += f"｜{f['traits']}"
            if f.get("recent"):
                line += f"——近况：{f['recent']}"
            lines.append(line)
        lines.append(
            "提到他们时要自然，像是刚跟他们联系过，不要念简介，"
            "也不要在一条消息里把所有人搬出来。"
            "不要虚构名单之外的新「重要人物」，路人可以一笔带过不提名字。",
        )
        text = "\n".join(lines)
        return text[: self.inject_max_chars]

    # ---------------- 钩子 ----------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        umo = self._umo(event)
        try:
            sp = getattr(req, "system_prompt", "") or ""
            if sp:
                self._persona_cache[umo] = sp[:2000]
        except Exception:  # noqa: BLE001
            pass

        if not self.enable_inject:
            return

        try:
            friends = await self._ensure_circle(umo)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Social] 准备社会链失败: {exc}")
            friends = self.state.get("friends") or []

        if not friends:
            return

        # 每天第一次对话时，后台推进近况（不阻塞本轮回复）
        if self.evolve_daily:
            today = datetime.now().strftime("%Y-%m-%d")
            if self.state.get("last_evolve") != today and not self._evolving:
                self._evolving = True

                async def _bg():
                    try:
                        await self._evolve()
                    finally:
                        self._evolving = False

                try:
                    asyncio.create_task(_bg())
                except Exception:  # noqa: BLE001
                    self._evolving = False

        hint = self._render(friends)
        try:
            from astrbot.core.agent.message import TextPart

            part = TextPart(text=hint)
            if hasattr(part, "mark_as_temp"):
                part = part.mark_as_temp()
            req.extra_user_content_parts.append(part)
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            req.system_prompt = f"{req.system_prompt or ''}\n{hint}"
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[Social] 注入失败: {exc}")

    # ---------------- 命令 ----------------

    @filter.command("社会链")
    async def cmd_show(self, event: AstrMessageEvent):
        """查看角色当前的社会关系"""
        umo = self._umo(event)
        friends = await self._ensure_circle(umo)
        if not friends:
            yield event.plain_result("还没有生成社会关系。")
            return
        lines = [f"共 {len(friends)} 个人："]
        for f in friends:
            lines.append(
                f"· {f['name']}（{f.get('relation') or '—'}）"
                f"｜{f.get('traits') or '—'}\n  近况：{f.get('recent') or '—'}",
            )
        gen = self.state.get("generated_at") or "未知"
        lines.append(f"\n生成于 {gen}，最近推进 {self.state.get('last_evolve') or '尚未'}")
        yield event.plain_result("\n".join(lines))

    @filter.command("社会链重写")
    async def cmd_regen(self, event: AstrMessageEvent):
        """按当前人设重新生成整套社会关系"""
        umo = self._umo(event)
        yield event.plain_result("正在重新想我身边都有谁…")
        friends = await self._ensure_circle(umo, force=True)
        names = "、".join(f"{f['name']}（{f.get('relation') or '—'}）" for f in friends)
        yield event.plain_result(f"好了，我身边有：{names}")

    @filter.command("社会链刷新")
    async def cmd_evolve(self, event: AstrMessageEvent):
        """手动推进一次近况（不等到第二天）"""
        if not (self.state.get("friends") or []):
            yield event.plain_result("还没有社会关系，先发 /社会链 生成一份。")
            return
        yield event.plain_result("看看他们最近怎么样了…")
        await self._evolve()
        friends = self.state.get("friends") or []
        lines = [f"· {f['name']}：{f.get('recent') or '—'}" for f in friends]
        yield event.plain_result("现在的近况：\n" + "\n".join(lines))
