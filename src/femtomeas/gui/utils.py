from dash import Input, Output

def make_scroll_callback(component_id, app):
    app.clientside_callback(
        f"""
        function(value) {{
            const ta = document.getElementById("{component_id}");
            if (!ta) return;

            const scrollBottom = () => {{
                ta.scrollTop = ta.scrollHeight;
                ta.scrollTo(0, ta.scrollHeight);
            }};

            // Wait for Dash/React to finish painting the new value
            requestAnimationFrame(() => {{
                requestAnimationFrame(() => {{
                    scrollBottom();

                    // One more pass in case the textarea layout settles a tick later
                    setTimeout(scrollBottom, 50);
                }});
            }});
        }}
        """,
        Input(component_id, "value"),
        prevent_initial_call=True,
    )

# def make_scroll_callback(component_id, app):
#     app.clientside_callback(
#         f"""
#         function(value) {{
#             const ta = document.getElementById("{component_id}");
#             if (!ta) return;

#             const scrollIfNeeded = () => {{
#                 const threshold = 20;
#                 const nearBottom =
#                     ta.scrollTop + ta.clientHeight >= ta.scrollHeight - threshold;

#                 if (nearBottom) {{
#                     ta.scrollTop = ta.scrollHeight;
#                     console.log("scrolled {component_id} to bottom");
#                 }}
#             }};

#             requestAnimationFrame(() => {{
#                 requestAnimationFrame(scrollIfNeeded);
#             }});
#         }}
#         """,
#         Input(component_id, "value"),
#         prevent_initial_call=True,
#     )


# def make_scroll_callback(component_id, app):
#     return app.clientside_callback(
#         f"""
#         function(value) {{
#             console.log("smartScroll fired for error-display");
#             const ta = document.getElementById('{component_id}');
#             if (!ta){{
#                console.log("smartScroll no component with that id");
#                return;
#             }}
#             setTimeout(() => {{
#                 const threshold = 20;
#                 const isNearBottom =
#                     ta.scrollTop + ta.clientHeight >= ta.scrollHeight - threshold;

#                 if (isNearBottom) {{
#                     ta.scrollTop = ta.scrollHeight;
#                     console.log("smartScroll is near bottom");
#                 }}else{{
#                     console.log("smartScroll not near bottom");
#                 }}
#             }}, 0);
#             return;
#         }}
#         """,
#         Input(component_id, "value"),
#         prevent_initial_call=True
#     )
