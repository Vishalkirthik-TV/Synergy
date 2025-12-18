import asyncio
import base64
from playwright.async_api import async_playwright, Page as AsyncPage, BrowserContext as AsyncBrowserContext
from playwright.sync_api import sync_playwright, Page as SyncPage, BrowserContext as SyncBrowserContext
import logging
import sys
import concurrent.futures

# CRITICAL FIX: Windows requires ProactorEventLoop for subprocess support (Playwright)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)

class BrowserManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.use_sync_api = (sys.platform == "win32")  # Use sync API on Windows
        
        # Async API attributes
        self.playwright_async = None
        self.browser_async = None
        self.context_async: AsyncBrowserContext = None
        self.page_async: AsyncPage = None
        
        # Sync API attributes (for Windows)
        self.playwright_sync = None
        self.browser_sync = None
        self.context_sync: SyncBrowserContext = None
        self.page_sync: SyncPage = None
        
        self.is_active = False
        self.lock = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if self.use_sync_api else None

    async def start_browser(self):
        """Initialize the browser if not already running"""
        async with self.lock:
            if self.is_active:
                return

            try:
                logger.info("🚀 Launching Headless Browser (Playwright)...")
                
                if self.use_sync_api:
                    # Windows: Use sync API in thread pool
                    await self._start_browser_sync()
                else:
                    # Linux/Mac: Use async API directly
                    await _start_browser_async()
                    
                self.is_active = True
                logger.info("✅ Browser started successfully")
            except Exception as e:
                logger.error(f"❌ Failed to start browser: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await self.stop_browser()
                raise
    
    async def _start_browser_async(self):
        """Standard async browser initialization"""
        self.playwright_async = await async_playwright().start()
        self.browser_async = await self.playwright_async.chromium.launch(headless=False)  # Visible browser
        self.context_async = await self.browser_async.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page_async = await self.context_async.new_page()
    
    async def _start_browser_sync(self):
        """Sync browser initialization for Windows (runs in thread pool)"""
        def _sync_start():
            self.playwright_sync = sync_playwright().start()
            self.browser_sync = self.playwright_sync.chromium.launch(headless=False)  # Visible browser
            self.context_sync = self.browser_sync.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self.page_sync = self.context_sync.new_page()
        
        await asyncio.get_event_loop().run_in_executor(self._executor, _sync_start)

    async def stop_browser(self):
        """Close browser resources"""
        async with self.lock:
            try:
                if self.use_sync_api:
                    def _sync_stop():
                        if self.page_sync:
                            self.page_sync.close()
                        if self.context_sync:
                            self.context_sync.close()
                        if self.browser_sync:
                            self.browser_sync.close()
                        if self.playwright_sync:
                            self.playwright_sync.stop()
                    
                    await asyncio.get_event_loop().run_in_executor(self._executor, _sync_stop)
                    self.page_sync = None
                    self.context_sync = None
                    self.browser_sync = None
                    self.playwright_sync = None
                else:
                    if self.page_async:
                        await self.page_async.close()
                    if self.context_async:
                        await self.context_async.close()
                    if self.browser_async:
                        await self.browser_async.close()
                    if self.playwright_async:
                        await self.playwright_async.stop()
                    
                    self.page_async = None
                    self.context_async = None
                    self.browser_async = None
                    self.playwright_async = None
                
                self.is_active = False
                logger.info("🛑 Browser stopped")
            except Exception as e:
                logger.error(f"Error stopping browser: {e}")

    async def navigate(self, url: str):
        if not self.is_active:
            await self.start_browser()
        
        logger.info(f"🌐 Navigating to: {url}")
        try:
            if self.use_sync_api:
                def _sync_navigate():
                    self.page_sync.goto(url, wait_until='domcontentloaded', timeout=15000)
                    import time
                    time.sleep(2)
                await asyncio.get_event_loop().run_in_executor(self._executor, _sync_navigate)
            else:
                await self.page_async.goto(url, wait_until='domcontentloaded', timeout=15000)
                await asyncio.sleep(2)
            return True, "Navigated successfully"
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return False, str(e)

    async def get_screenshot(self) -> str:
        """Capture screenshot and return as base64 string"""
        if not self.is_active:
            return None
        
        page = self.page_sync if self.use_sync_api else self.page_async
        if not page:
            return None
        
        try:
            if self.use_sync_api:
                screenshot_bytes = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.page_sync.screenshot(type='jpeg', quality=70)
                )
            else:
                screenshot_bytes = await self.page_async.screenshot(type='jpeg', quality=70)
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return None

    async def click_element(self, selector: str):
        if not self.is_active:
            return False, "Browser not active"
        
        page = self.page_sync if self.use_sync_api else self.page_async
        if not page:
            return False, "Browser not active"
        
        try:
            logger.info(f"🖱️ Clicking: {selector}")
            if self.use_sync_api:
                def _sync_click():
                    self.page_sync.click(selector, timeout=5000)
                    import time
                    time.sleep(1)
                await asyncio.get_event_loop().run_in_executor(self._executor, _sync_click)
            else:
                await self.page_async.click(selector, timeout=5000)
                await asyncio.sleep(1)
            return True, "Clicked successfully"
        except Exception as e:
            logger.error(f"Click error: {e}")
            return False, str(e)

    async def type_text(self, selector: str, text: str):
        if not self.is_active:
            return False, "Browser not active"
        
        page = self.page_sync if self.use_sync_api else self.page_async
        if not page:
            return False, "Browser not active"
            
        try:
            logger.info(f"⌨️ Typing '{text}' into {selector}")
            if self.use_sync_api:
                def _sync_fill():
                    self.page_sync.fill(selector, text)
                    import time
                    time.sleep(0.5)
                await asyncio.get_event_loop().run_in_executor(self._executor, _sync_fill)
            else:
                await self.page_async.fill(selector, text)
                await asyncio.sleep(0.5)
            return True, "Typed successfully"
        except Exception as e:
            logger.error(f"Type error: {e}")
            return False, str(e)
            
    async def press_key(self, key: str):
        if not self.is_active:
             return False, "Browser not active"
        
        page = self.page_sync if self.use_sync_api else self.page_async
        if not page:
            return False, "Browser not active"
        try:
            if self.use_sync_api:
                def _sync_press():
                    self.page_sync.keyboard.press(key)
                    import time
                    time.sleep(1)
                await asyncio.get_event_loop().run_in_executor(self._executor, _sync_press)
            else:
                await self.page_async.keyboard.press(key)
                await asyncio.sleep(1)
            return True, f"Pressed {key}"
        except Exception as e:
            return False, str(e)
            
    async def run_action(self, action_type: str, **kwargs):
        """Generic entry point for actions"""
        if action_type == "navigate":
            return await self.navigate(kwargs.get('url'))
        elif action_type == "click":
            return await self.click_element(kwargs.get('selector'))
        elif action_type == "type":
            return await self.type_text(kwargs.get('selector'), kwargs.get('text'))
        elif action_type == "press":
            return await self.press_key(kwargs.get('key'))
        elif action_type == "screenshot":
            return await self.get_screenshot()
        elif action_type == "scroll_down":
            page = self.page_sync if self.use_sync_api else self.page_async
            if page:
                if self.use_sync_api:
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: self.page_sync.evaluate("window.scrollBy(0, 500)")
                    )
                else:
                    await self.page_async.evaluate("window.scrollBy(0, 500)")
                return True, "Scrolled down"
        elif action_type == "scroll_up":
            page = self.page_sync if self.use_sync_api else self.page_async
            if page:
                if self.use_sync_api:
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: self.page_sync.evaluate("window.scrollBy(0, -500)")
                    )
                else:
                    await self.page_async.evaluate("window.scrollBy(0, -500)")
                return True, "Scrolled up"
        
        return False, "Unknown action"
