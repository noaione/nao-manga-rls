# [nao] Manga Release Scripts

This repo contains stuff that I use to release and collect manga from a certain cat website.

All my release use the `[nao]` tag.

## Released Stuff
- 5 Seconds Before a Witch Falls in Love (Finished)
- Accomplishments of the Duke's Daughter (v06+) (Ongoing)
- Chillin' in Another World with Level 2 Super Cheat Powers (v02+) (Ongoing)
- Chronicles of an Aristocrat Reborn in Another World (Ongoing)
- DNA Doesn't Tell Us (Finished)
- Drugstore in Another World: The Slow Life of a Cheat Pharmacist (Ongoing)
- I've Been Killing Slimes for 300 Years and Maxed Out My Level (v08+) (Ongoing)
- Kuma Kuma Kuma Bear (Ongoing)
- Magic Artisan Dahlia Wilts No More (Ongoing)
- My Next Life as a Villainess: All Routes Lead to Doom! (Ongoing)
- Nameless Asterism (Finished)
- Necromance (v02+) (Ongoing)
- New Game! (Ongoing)
- The Dangers in My Heart (v04+) (Ongoing)
- The Invicible Shovel (Ongoing)
- The Savior's Book Café Story in Another World (Ongoing)
- Yuri is My Job! (Ongoing)

**Weekly-ish Release**
- The Necromancer Maid
- The Villainess Who Became Nightingale

**Planned**
- Fushi no Kami
- ~~I Was a Bottom-Tier Bureaucrat for 1,500 Years, and the Demon King Made Me a Minister (Final Volume)~~
- The Apothecary Diaries (v05+)
- ~~Welcome to Japan, Ms. Elf! (v04+)~~

**Dropped**
- BAKEMONOGATARI (v13)
- Chainsaw Man (v09)
- Medaka Kuroiwa is Impervious to My Charms (v01)
- My Deer Friend Nekotan (v01)
- The Ice-Guy and His Cool Female Colleague (Taken over by someone else)

Anything in Dropped means I already ripped it once, but I dropped it.

You can find it on a certain cat website.

## Requirements
- Python 3.7+
- imagemagick (for `level` and `spreads` command)
- exiftool (for `releases`, optional)

## Scripts
This repo also have a module or script to release and split stuff up.

To install, you need git since I'm not publishing this to PyPi.
After that you can run this:

```sh
pip install -U git+https://github.com/noaione/nao-manga-rls.git@module-rewrite#egg=nmanga
```

This will install this project which then you can use the command `nmanga` to execute everything.

### Commands

#### `autosplit`
Same as the old auto split script, this will ask a series of question where you can automatically split the content of a volume archive into a chapters archives (cbz format).

It will ask for the file `Manga Title` then the `Publisher` (if there is chapter title).

The regex should match something like this:
- `Manga Title - c001 (v01) - p001 [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg`
- `Manga Title - c001 (v01) - p000 [Cover] [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg`
- `Manga Title - c001 (v01) - p000 [Cover] [dig] [Publisher Name?] [Group] {HQ}.jpg`
- `Manga Title - c001 (v01) - p001 [dig] [Publisher Name?] [Group] {HQ}.jpg`

This should match most of the release from a certain cat website (`danke`, `LuCaZ`, `1r0n`, etc.)

#### `level`
Automatically batch color level an archive or folder containing images.<br />
This can be useful to some VIZ release since sometimes the "black" color is not really that "black" (ex. `#202020` instead of `#000000`)
