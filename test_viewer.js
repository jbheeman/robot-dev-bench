const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({headless: true});
  const page = await browser.newPage();
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
  await page.goto('http://127.0.0.1:8000');
  
  // Wait for URDF to load
  await page.waitForTimeout(2000);
  
  // Mock upload response
  await page.evaluate(() => {
    // Generate dummy data
    const ticks = [];
    const q = [];
    for(let i=0; i<300; i++) {
      ticks.push(i * 33.3);
      q.push(new Array(29).fill(0));
    }
    window.loadPlaybackData({ticks: ticks, q: q});
  });
  
  console.log("Mock data loaded. Clicking play...");
  await page.evaluate(() => document.getElementById('play-btn').click());
  
  await page.waitForTimeout(2000);
  
  const currentFrame = await page.evaluate(() => document.getElementById('timeline').value);
  console.log("Timeline value after 2 seconds:", currentFrame);
  
  await browser.close();
})();
