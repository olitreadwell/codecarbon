import { describe, expect, it } from "vitest";

import { pickTimeFormat } from "@/helpers/time-axis";

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

describe("pickTimeFormat", () => {
    it("shows seconds for spans under two minutes", () => {
        expect(pickTimeFormat(30_000)).toBe("HH:mm:ss");
    });

    it("shows time only for spans under a day", () => {
        expect(pickTimeFormat(2 * HOUR)).toBe("HH:mm");
    });

    it("shows date and time for spans under a week", () => {
        expect(pickTimeFormat(3 * DAY)).toBe("MMM d, HH:mm");
    });

    it("shows date and time for multi-week spans so same-day runs stay distinguishable", () => {
        expect(pickTimeFormat(10 * DAY)).toBe("MMM d, HH:mm");
    });

    it("shows month and year for spans over a year", () => {
        expect(pickTimeFormat(400 * DAY)).toBe("MMM yyyy");
    });
});
