/**
 * @file   PlaceDrawer.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef DREAMPLACE_PLACEDRAWER_H
#define DREAMPLACE_PLACEDRAWER_H

#include <fstream>
#include <ostream>
#include <set>
#include <string>

#if DRAWPLACE == 1
#include <cairo-pdf.h>
#include <cairo-ps.h>
#include <cairo-svg.h>
#include <cairo.h>
#endif

#include <limbo/parsers/gdsii/stream/GdsWriter.h>
#include <cstdio>
#include <cstdlib>

#include "utility/src/utils.h"

typedef struct _cairo_surface cairo_surface_t;

DREAMPLACE_BEGIN_NAMESPACE

/// PlaceDrawer write files in various formats
template <typename T, typename I>
class PlaceDrawer {
 public:
  typedef T coordinate_type;
  typedef I index_type;

  enum FileFormat {
    EPS = 0,  // handle by cairo
    PDF = 1,  // handle by cairo
    SVG = 2,  // handle by cairo
    PNG = 3,  // handle by cairo
    GDSII = 4
  };
  enum DrawContent {
    NONE = 0,
    NODE = 1,
    NODETEXT = 2,
    PIN = 4,
    NET = 8,
    ALL = NODE | NODETEXT | PIN | NET
  };
  /// constructor
  PlaceDrawer(
      const coordinate_type* x, const coordinate_type* y,
      const coordinate_type* node_size_x, const coordinate_type* node_size_y,
      const coordinate_type* pin_offset_x, const coordinate_type* pin_offset_y,
      const index_type* pin2node_map, const index_type num_nodes,
      const index_type num_movable_nodes, const index_type num_filler_nodes,
      const index_type num_pins, const coordinate_type xl,
      const coordinate_type yl, const coordinate_type xh,
      const coordinate_type yh, const coordinate_type site_width,
      const coordinate_type row_height, const coordinate_type bin_size_x,
      const coordinate_type bin_size_y, int content = ALL)
      : m_x(x),
        m_y(y),
        m_node_size_x(node_size_x),
        m_node_size_y(node_size_y),
        m_pin_offset_x(pin_offset_x),
        m_pin_offset_y(pin_offset_y),
        m_pin2node_map(pin2node_map),
        m_num_nodes(num_nodes),
        m_num_movable_nodes(num_movable_nodes),
        m_num_filler_nodes(num_filler_nodes),
        m_num_pins(num_pins),
        m_xl(xl),
        m_yl(yl),
        m_xh(xh),
        m_yh(yh),
        m_site_width(site_width),
        m_row_height(row_height),
        m_bin_size_x(bin_size_x),
        m_bin_size_y(bin_size_y),
        m_content(content) {}

  bool run(std::string const& filename, FileFormat ff) const {
    dreamplacePrint(kINFO, "writing placement to %s\n", filename.c_str());
    bool flag = false;

    // PlaceDB const& placeDB = m_db.placeDB();
    int width, height; 
    if (m_xh - m_xl < m_yh - m_yl) {
      height = 800; 
      width = round(height * (double)(m_xh - m_xl) / (m_yh - m_yl)); 
    } else {
      width = 800; 
      height = round(width * (double)(m_yh - m_yl) / (m_xh - m_xl));
    }

    switch (ff) {
      case EPS:
      case PDF:
      case SVG:
      case PNG:
        flag = writeFig(filename.c_str(), width, height, ff);
        break;
      case GDSII:
        flag = writeGdsii(filename);
        break;
      default:
        dreamplacePrint(kERROR, "unknown writing format at line %u\n",
                        __LINE__);
        break;
    }

    return flag;
  }

  /// \param first and last mark nodes whose nets will be drawn
  template <typename Iterator>
  bool run(std::string const& filename, FileFormat ff, Iterator first,
           Iterator last) {
    m_sMarkNode.insert(first, last);
    bool flag = run(filename, ff);
    m_sMarkNode.clear();
    return flag;
  }

 protected:
  /// write formats supported by cairo
  /// \param width of screen
  /// \param height of screen
  void paintCairo(cairo_surface_t* cs, double width, double height) const {
#if DRAWPLACE == 1
    double ratio[2] = {width / (m_xh - m_xl), height / (m_yh - m_yl)};
    char buf[16];
    cairo_t* c;
    cairo_text_extents_t extents;

    c = cairo_create(cs);
    cairo_save(c);  // save status
    cairo_translate(c, 0 - m_xl * ratio[0],
                    height + m_yl * ratio[1]);  // translate is additive
    cairo_scale(c, ratio[0], -ratio[1]);        // scale is additive

    // background
    cairo_rectangle(c, m_xl, m_yl, (m_xh - m_xl), (m_yh - m_yl));
    cairo_set_source_rgb(c, 1.0, 1.0, 1.0);
    cairo_fill(c);
    cairo_rectangle(c, m_xl, m_yl, (m_xh - m_xl), (m_yh - m_yl));
    cairo_set_line_width(c, 0.001);
    cairo_set_source_rgb(c, 0.1, 0.1, 0.1);
    cairo_stroke(c);

    // bins
    cairo_set_line_width(c, 0.001);
    cairo_set_source_rgba(c, 0.1, 0.1, 0.1, 0.8);
    for (coordinate_type bx = m_xl; bx < m_xh; bx += m_bin_size_x) {
      cairo_move_to(c, bx, m_yl);
      cairo_line_to(c, bx, m_yh);
      cairo_stroke(c);
    }
    for (coordinate_type by = m_yl; by < m_yh; by += m_bin_size_y) {
      cairo_move_to(c, m_xl, by);
      cairo_line_to(c, m_xh, by);
      cairo_stroke(c);
    }

    // nodes
    cairo_set_line_width(c, 0.001);
    cairo_select_font_face(c, "Sans", CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    if (m_content & NODE) {
      // fixed macro
      for (int i = m_num_movable_nodes; i < m_num_nodes - m_num_filler_nodes;
           ++i) {
        cairo_rectangle(c, m_x[i], m_y[i], m_node_size_x[i], m_node_size_y[i]);
        cairo_set_source_rgba(c, 1.0, 0.0, 0.0, 0.5);
        cairo_fill(c);
        cairo_rectangle(c, m_x[i], m_y[i], m_node_size_x[i], m_node_size_y[i]);
        cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
        cairo_stroke(c);
        if (m_content & NODETEXT) {
          sprintf(buf, "%u", i);
          cairo_set_font_size(c, m_node_size_y[i] / 20);
          cairo_text_extents(c, buf, &extents);
          cairo_move_to(c,
                        (m_x[i] + m_node_size_x[i] / 2) -
                            (extents.width / 2 + extents.x_bearing),
                        (m_y[i] + m_node_size_y[i] / 2) -
                            (extents.height / 2 + extents.y_bearing));
          cairo_show_text(c, buf);
        }
      }
      // filler
      for (int i = m_num_nodes - m_num_filler_nodes; i < m_num_nodes; ++i) {
        cairo_rectangle(c, m_x[i], m_y[i], m_node_size_x[i], m_node_size_y[i]);
        cairo_set_source_rgba(c, 115 / 255.0, 115 / 255.0, 125 / 255.0, 0.5);
        cairo_fill(c);
        cairo_rectangle(c, m_x[i], m_y[i], m_node_size_x[i], m_node_size_y[i]);
        cairo_set_source_rgba(c, 100 / 255.0, 100 / 255.0, 100 / 255.0, 0.8);
        cairo_stroke(c);
        if (m_content & NODETEXT) {
          sprintf(buf, "%u", i);
          cairo_set_font_size(c, m_node_size_y[i] / 20);
          cairo_text_extents(c, buf, &extents);
          cairo_move_to(c,
                        (m_x[i] + m_node_size_x[i] / 2) -
                            (extents.width / 2 + extents.x_bearing),
                        (m_y[i] + m_node_size_y[i] / 2) -
                            (extents.height / 2 + extents.y_bearing));
          cairo_show_text(c, buf);
        }
      }
      // movable
      for (int i = 0; i < m_num_movable_nodes; ++i) {
        cairo_rectangle(c, m_x[i], m_y[i], m_node_size_x[i], m_node_size_y[i]);
        cairo_set_source_rgba(c, 0, 0, 1, 0.5);
        cairo_fill(c);
        cairo_rectangle(c, m_x[i], m_y[i], m_node_size_x[i], m_node_size_y[i]);
        cairo_set_source_rgba(c, 0, 0, 0.8, 0.8);
        cairo_stroke(c);
        if (m_content & NODETEXT) {
          sprintf(buf, "%u", i);
          cairo_set_font_size(c, m_node_size_y[i] / 20);
          cairo_text_extents(c, buf, &extents);
          cairo_move_to(c,
                        (m_x[i] + m_node_size_x[i] / 2) -
                            (extents.width / 2 + extents.x_bearing),
                        (m_y[i] + m_node_size_y[i] / 2) -
                            (extents.height / 2 + extents.y_bearing));
          cairo_show_text(c, buf);
        }
      }
    }
    cairo_restore(c);

    cairo_show_page(c);

    cairo_destroy(c);
#else
    dreamplacePrint(kWARN,
                    "cs = %p, width = %g, height = %g are not used, as "
                    "DRAWPLACE not enabled\n",
                    cs, width, height);
#endif
  }
  bool writeFig(const char* fname, double width, double height,
                FileFormat ff) const {
#if DRAWPLACE == 1
    cairo_surface_t* cs;

    switch (ff) {
      case PNG:
        cs = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
        break;
      case PDF:
        cs = cairo_pdf_surface_create(fname, width, height);
        break;
      case EPS:
        cs = cairo_ps_surface_create(fname, width, height);
        break;
      case SVG:
        cs = cairo_svg_surface_create(fname, width, height);
        break;
      default:
        dreamplacePrint(kERROR, "unknown file format in %s\n", __func__);
        return false;
    }

    paintCairo(cs, width, height);

    cairo_surface_flush(cs);
    // need additional writing call for PNG
    if (ff == PNG) cairo_surface_write_to_png(cs, fname);
    cairo_surface_destroy(cs);
    return true;
#else
    dreamplacePrint(kWARN,
                    "filename = %s, width = %g, height = %g, file format = %d "
                    "not used, as DRAWPLACE not enabled\n",
                    fname, width, height, (int)ff);
    return false;
#endif
  }
  /// scale source coordinate to target screen
  double scaleToScreen(double coord, double srcOffset, double srcSize,
                       double tgtOffset, double tgtSize) const {
    double ratio = tgtSize / srcSize;
    return tgtOffset + (coord - srcOffset) * ratio;
  }

  /// write gdsii format
  virtual bool writeGdsii(std::string const& filename) const {
    double scale_rato = 1000;
    GdsParser::GdsWriter gw(filename.c_str());
    gw.create_lib("TOP", 0.001, 1e-6 / scale_rato);
    gw.gds_write_bgnstr();
    gw.gds_write_strname("TOP");

    // kernel function to fill in contents
    writeGdsiiContent(gw, scale_rato);

    gw.gds_write_endstr();
    gw.gds_write_endlib();

    return true;
  }
  /// write contents to GDSII
  virtual void writeGdsiiContent(GdsParser::GdsWriter& gw,
                                 double scale_rato) const {
    // layer specification
    // it is better to use even layers, because text appears on odd layers
    const unsigned dieAreaLayer = getLayer(true);
    const unsigned rowLayer = getLayer(false);
    const unsigned subRowLayer = getLayer(false);
    const unsigned binRowLayer = getLayer(false);
    const unsigned binLayer = getLayer(false);
    const unsigned sbinLayer = getLayer(false);
    const unsigned movableCellBboxLayer = getLayer(false);
    const unsigned fixedCellBboxLayer = getLayer(false);
    const unsigned blockageBboxLayer = getLayer(false);
    const unsigned fillerCellBboxLayer = getLayer(false);
    const unsigned pinLayer = getLayer(false);
    const unsigned multiRowCellBboxLayer = getLayer(false);
    const unsigned movePathLayer = getLayer(false);
    const unsigned markedNodeLayer = getLayer(false);  // together with netLayer
    const unsigned netLayer = getLayer(false);

    dreamplacePrint(
        kINFO,
        "Layer: dieArea:%u, row:%u, subRow:%u, binRow:%u, bin:%u, sbin:%u, "
        "movableCellBbox:%u, fixedCellBbox:%u, blockageBbox:%u, "
        "fillerCellBboxLayer:%u, pin:%u, multiRowCellBbox:%u, "
        "movePathLayer:%u, markedNodeLayer:%u, net:from %u\n",
        dieAreaLayer, rowLayer, subRowLayer, binRowLayer, binLayer, sbinLayer,
        movableCellBboxLayer, fixedCellBboxLayer, blockageBboxLayer,
        fillerCellBboxLayer, pinLayer, multiRowCellBboxLayer, movePathLayer,
        markedNodeLayer, netLayer);

    char buf[1024];

    // write dieArea
    gw.write_box(dieAreaLayer, 0, m_xl * scale_rato, m_yl * scale_rato,
                 m_xh * scale_rato, m_yh * scale_rato);
    // write bins
    for (coordinate_type bx = m_xl; bx < m_xh; bx += m_bin_size_x) {
      for (coordinate_type by = m_yl; by < m_yh; by += m_bin_size_y) {
        coordinate_type bxl = bx;
        coordinate_type byl = by;
        coordinate_type bxh = std::min(bxl + m_bin_size_x, m_xh);
        coordinate_type byh = std::min(byl + m_bin_size_y, m_yh);
        gw.write_box(binLayer, 0, bxl * scale_rato, byl * scale_rato,
                     bxh * scale_rato, byh * scale_rato);
        dreamplaceSPrint(kNONE, buf, "%u,%u",
                         (unsigned int)round((bx - m_xl) / m_bin_size_x),
                         (unsigned int)round((by - m_yl) / m_bin_size_y));
        gw.gds_create_text(buf, (bxl + bxh) / 2 * scale_rato,
                           (byl + byh) / 2 * scale_rato, binLayer + 1, 5);
      }
    }
    // write cells
    for (index_type i = 0; i < m_num_nodes; ++i) {
      // bounding box of cells and its name
      coordinate_type node_xl = m_x[i];
      coordinate_type node_yl = m_y[i];
      coordinate_type node_xh = node_xl + m_node_size_x[i];
      coordinate_type node_yh = node_yl + m_node_size_y[i];
      unsigned layer;
      if (i < m_num_movable_nodes)  // movable cell
      {
        layer = movableCellBboxLayer;
      } else if (i >= m_num_nodes - m_num_filler_nodes)  // filler cell
      {
        layer = fillerCellBboxLayer;
      } else  // fixed cells
      {
        layer = fixedCellBboxLayer;
      }

      if (layer == fixedCellBboxLayer ||
          m_sMarkNode.empty())  // do not write cells if there are marked cells
      {
        gw.write_box(layer, 0, node_xl * scale_rato, node_yl * scale_rato,
                     node_xh * scale_rato, node_yh * scale_rato);
        dreamplaceSPrint(kNONE, buf, "(%u)%s", i, getTextOnNode(i).c_str());
        gw.gds_create_text(buf, (node_xl + node_xh) / 2 * scale_rato,
                           (node_yl + node_yh) / 2 * scale_rato, layer + 1, 5);

        if (i < m_num_movable_nodes &&
            m_node_size_y[i] > m_row_height)  // multi-row cell
        {
          gw.write_box(multiRowCellBboxLayer, 0, node_xl * scale_rato,
                       node_yl * scale_rato, node_xh * scale_rato,
                       node_yh * scale_rato);
          gw.gds_create_text(buf, (node_xl + node_xh) / 2 * scale_rato,
                             (node_yl + node_yh) / 2 * scale_rato,
                             multiRowCellBboxLayer + 1, 5);
        }
      }
      if (m_sMarkNode.count(i))  // highlight marked nodes
      {
        gw.write_box(markedNodeLayer, 0, node_xl * scale_rato,
                     node_yl * scale_rato, node_xh * scale_rato,
                     node_yh * scale_rato);
        dreamplaceSPrint(kNONE, buf, "(%u)%s", i, getTextOnNode(i).c_str());
        gw.gds_create_text(buf, (node_xl + node_xh) / 2 * scale_rato,
                           (node_yl + node_yh) / 2 * scale_rato,
                           markedNodeLayer + 1, 5);
      }
    }
    // write pins
    for (index_type i = 0; i < m_num_pins; ++i) {
      coordinate_type pin_xl;
      coordinate_type pin_yl;
      coordinate_type pin_xh;
      coordinate_type pin_yh;
      getPinBbox(i, scale_rato, pin_xl, pin_yl, pin_xh, pin_yh);
      // bounding box of pins and its macropin name
      gw.write_box(pinLayer, 0, pin_xl * scale_rato, pin_yl * scale_rato,
                   pin_xh * scale_rato, pin_yh * scale_rato);
      gw.gds_create_text(getTextOnPin(i).c_str(),
                         (pin_xl + pin_xh) / 2 * scale_rato,
                         (pin_yl + pin_yh) / 2 * scale_rato, pinLayer + 1, 5);
    }
  }
  /// automatically increment by 2
  /// \param reset controls whehter restart from 1
  unsigned getLayer(bool reset = false) const {
    static unsigned count = 0;
    if (reset) count = 0;
    return (++count) << 1;
  }
  /// \param i node id
  /// \return text to be shown on cell
  std::string getTextOnNode(index_type i) const { return ""; }
  /// \param i pin id
  /// \return text to be shown on pin
  std::string getTextOnPin(index_type i) const { return "NA"; }
  /// \brief set pin bounding box
  /// \param i pin id
  void getPinBbox(index_type i, double scale_rato, coordinate_type& xl,
                  coordinate_type& yl, coordinate_type& xh,
                  coordinate_type& yh) const {
    index_type node_id = m_pin2node_map[i];
    coordinate_type x = m_x[node_id];
    coordinate_type y = m_y[node_id];
    coordinate_type offset_x = m_pin_offset_x[i];
    coordinate_type offset_y = m_pin_offset_y[i];
    coordinate_type pin_size =
        std::max(std::min(m_site_width, m_row_height) / 10,
                 (coordinate_type)(1.0 / scale_rato));
    xl = x + offset_x - pin_size;
    yl = y + offset_y - pin_size;
    xh = x + offset_x + pin_size;
    yh = y + offset_y + pin_size;
  }

  const coordinate_type* m_x;
  const coordinate_type* m_y;
  const coordinate_type* m_node_size_x;
  const coordinate_type* m_node_size_y;
  const coordinate_type* m_pin_offset_x;
  const coordinate_type* m_pin_offset_y;
  const index_type* m_pin2node_map;
  index_type m_num_nodes;
  index_type m_num_movable_nodes;
  index_type m_num_filler_nodes;
  index_type m_num_pins;
  coordinate_type m_xl;
  coordinate_type m_yl;
  coordinate_type m_xh;
  coordinate_type m_yh;
  coordinate_type m_site_width;
  coordinate_type m_row_height;
  coordinate_type m_bin_size_x;
  coordinate_type m_bin_size_y;
  std::set<index_type> m_sMarkNode;  ///< marked nodes whose net will be drawn
  int m_content;                     ///< content for DrawContent
};

DREAMPLACE_END_NAMESPACE

#endif
