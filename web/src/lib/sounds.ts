// Local sound effects from public/audio (git-ignored; see its README.md).
// Missing files are fine: play() just fails quietly.

const BASE = import.meta.env.BASE_URL;

/** Headline score at or above which the Giga Chad theme plays. */
export const MOG_THRESHOLD = 8.0;

export type Sounds = {
  /** Play or pause the theme. Pausing keeps the position, so it resumes, not restarts. */
  setMusic(playing: boolean): void;
  boom(): void;
  stop(): void;
};

export function createSounds(): Sounds {
  const music = new Audio(`${BASE}audio/gigachad.mp3`);
  music.loop = true;
  music.preload = "auto";
  const boom = new Audio(`${BASE}audio/vine-boom.mp3`);
  boom.preload = "auto";
  // Browsers block audio until the user has clicked on the page; ignore those
  // and missing-file errors.
  const quietly = (p: Promise<void>) => p.catch(() => undefined);

  return {
    setMusic(playing) {
      if (playing && music.paused) quietly(music.play());
      else if (!playing && !music.paused) music.pause();
    },
    boom() {
      boom.currentTime = 0;
      quietly(boom.play());
    },
    stop() {
      music.pause();
    },
  };
}
