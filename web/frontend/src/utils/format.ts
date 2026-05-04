const WINDOW_PATTERN = /^window_(\d+)h_to_(\d+)h$/;
const START_PATTERN = /^start_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})$/;

function hoursToLabel(hours: number): string {
  if (hours % 24 === 0) {
    return `${hours / 24} 天`;
  }
  return `${hours}h`;
}

export function formatWindow(name: string): string {
  const match = WINDOW_PATTERN.exec(name);
  if (!match) {
    return name;
  }
  const input = Number(match[1]);
  const output = Number(match[2]);
  return `${hoursToLabel(input)}历史 → ${hoursToLabel(output)}预测 (${input}h→${output}h)`;
}

export function formatStart(name: string): string {
  const match = START_PATTERN.exec(name);
  if (!match) {
    return name;
  }
  const [, year, month, day, hour, minute] = match;
  return `${year}-${month}-${day} ${hour}:${minute} (北京时间)`;
}
