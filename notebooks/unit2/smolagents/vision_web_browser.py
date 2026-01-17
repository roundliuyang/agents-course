import argparse
from io import BytesIO
from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, WebSearchTool, tool
from smolagents.agents import ActionStep
from smolagents.cli import load_model


alfred_guest_list_request = """
I am Alfred, the butler of Wayne Manor, responsible for verifying the identity of guests at party. A superhero has arrived at the entrance claiming to be Wonder Woman, but I need to confirm if she is who she says she is.
Please search for images of Wonder Woman and generate a detailed visual description based on those images. Additionally, navigate to Wikipedia to gather key details about her appearance. With this information, I can determine whether to grant her access to the event.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=alfred_guest_list_request,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    return parser.parse_args()


# 此函数作为 step_callback 传递给智能体，因为它在智能体执行的每一步结束时被触发。这使得智能体能够在整个过程中动态捕获和存储屏幕截图
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)   # 等待1秒，确保JavaScript动画完成后再截图
    driver = helium.get_driver()  # 获取当前浏览器驱动
    current_step = memory_step.step_number  # 获取当前浏览器驱动
    if driver is not None:    # 检查驱动是否存在
        # 遍历智能体记忆步骤，清理过旧的截图以节省内存
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            # 移除两步之前的截图，保持内存使用效率
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None   # 清空旧截图
        png_bytes = driver.get_screenshot_as_png()    # 获取浏览器截图的字节数据
        image = Image.open(BytesIO(png_bytes))   # 将字节数据转换为图像对象
        print(f"Captured a browser screenshot: {image.size} pixels")
        # 创建图像副本并存储到当前步骤的观察图像列表中（重要：创建副本确保持久性）
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # 使用当前URL更新观察信息
    url_info = f"Current url: {driver.current_url}"
    # 如果观察信息为空则直接赋值，否则追加到现有信息末尾
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return

# 定义一个工具函数，用于在页面上搜索指定文本并定位到第n个匹配项
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    通过XPath在当前页面搜索文本，模拟Ctrl+F功能并跳转到第n个匹配项
    Args:
        text: 要搜索的文本内容
        nth_result: 要跳转到的匹配项序号（默认为第1个）
    """
    # 使用XPath查找包含指定文本的所有元素
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    # 检查请求的匹配项序号是否存在
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    # 构造找到匹配项的结果信息
    result = f"Found {len(elements)} matches for '{text}'."
    # 获取指定序号的元素（数组索引从0开始所以减1）
    elem = elements[nth_result - 1]
    # 执行JavaScript使目标元素滚动到视窗可见区域
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    # 更新结果信息，说明已定位到哪个元素
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result

# 定义返回上一页的工具函数
@tool
def go_back() -> None:
    """浏览器后退功能，返回到上一个页面"""
    driver.back()

# 定义关闭弹窗的工具函数
@tool
def close_popups() -> str:
    """
    关闭页面上可见的模态框或弹出窗口
    注意：此功能不适用于cookie同意横幅
    """
    # 通过发送ESC按键来关闭弹窗
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>
You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>
If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>
If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.
To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>
You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>
If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""


def main():
    # Load environment variables
    # For example to use an OpenAI model, create a local .env file with OPENAI_API_KEY="<your_open_ai_key_here>"
    load_dotenv() 

    # Parse command line arguments
    args = parse_arguments()

    # Initialize the model based on the provided arguments
    model = load_model(args.model_type, args.model_id)

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(args.prompt + helium_instructions)


if __name__ == "__main__":
    main()