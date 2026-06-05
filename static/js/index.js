$(document).ready(function() {
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  if (document.querySelector('.carousel')) {
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
    });
  }

  if (window.bulmaSlider) {
    bulmaSlider.attach();
  }

  $(".multi-compare").each(function() {
    var compare = this;
    var ranges = Array.from(compare.querySelectorAll(".multi-range"));
    var minGap = 6;
    var activeIndex = 1;

    function clampHandles(activeRange) {
      var values = ranges.map(function(range) {
        return Number(range.value);
      });
      var index = ranges.indexOf(activeRange);

      if (index === 0) {
        values[0] = Math.min(values[0], values[1] - minGap);
      } else if (index === 1) {
        values[1] = Math.max(values[0] + minGap, Math.min(values[1], values[2] - minGap));
      } else if (index === 2) {
        values[2] = Math.max(values[2], values[1] + minGap);
      }

      values[0] = Math.max(5, Math.min(values[0], 95 - minGap * 2));
      values[1] = Math.max(values[0] + minGap, Math.min(values[1], 95 - minGap));
      values[2] = Math.max(values[1] + minGap, Math.min(values[2], 95));

      ranges.forEach(function(range, i) {
        range.value = values[i];
        compare.style.setProperty("--p" + (i + 1), values[i] + "%");
      });
    }

    function setHandleFromPointer(event) {
      var rect = compare.getBoundingClientRect();
      var percent = Math.max(5, Math.min(95, (event.clientX - rect.left) / rect.width * 100));
      ranges[activeIndex].value = percent;
      clampHandles(ranges[activeIndex]);
    }

    ranges.forEach(function(range) {
      range.addEventListener("input", function() {
        clampHandles(range);
      });
    });

    compare.addEventListener("pointerdown", function(event) {
      var rect = compare.getBoundingClientRect();
      var percent = Math.max(5, Math.min(95, (event.clientX - rect.left) / rect.width * 100));
      var values = ranges.map(function(range) {
        return Number(range.value);
      });

      activeIndex = values.reduce(function(bestIndex, value, index) {
        return Math.abs(value - percent) < Math.abs(values[bestIndex] - percent) ? index : bestIndex;
      }, 0);

      compare.setPointerCapture(event.pointerId);
      setHandleFromPointer(event);
    });

    compare.addEventListener("pointermove", function(event) {
      if (compare.hasPointerCapture(event.pointerId)) {
        setHandleFromPointer(event);
      }
    });

    compare.addEventListener("pointerup", function(event) {
      if (compare.hasPointerCapture(event.pointerId)) {
        compare.releasePointerCapture(event.pointerId);
      }
    });

    clampHandles(ranges[1]);
  });
});
